import copy
import cPickle as pickle
import math
import os
import pwd
import signal
import sys
import time
from collections import namedtuple
import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

import numpy
from sympy import Symbol
from sympy.core import basic
from sympy.printing.ccode import CCodePrinter
from sympy.printing.precedence import precedence

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.tools

from mako.lookup import TemplateLookup

import output

# For backwards-compatibility.  New scripts should import these functions
# directly from the output module.
from output import drift_velocity, abs_drift_velocity, diffusion_coefficient, avg_moments

PeriodInfo = namedtuple('PeriodInfo', 'period freq')
OutputDecl = namedtuple('OutputDecl', 'func vars')

# Map RNG name to number of uint state variables.
RNG_STATE = {
    'xs32': 1,
    'kiss32': 4,
    'nr32': 4,
}

want_save = False
want_dump = False
want_exit = False

def _sighandler(signum, frame):
    global want_dump, want_exit, want_save

    if signum == signal.SIGUSR2:
        want_save = True
    else:
        want_dump = True

        if signum in [signal.SIGINT, signal.SIGQUIT, signal.SIGTERM]:
            want_exit = True

def _convert_to_double(src):
    import re
    s = re.sub(r'float([^\.])', r'double\1', src)
    s = s.replace('logf(', 'log(')
    s = s.replace('expf(', 'exp(')
    s = s.replace('powf(', 'pow(')
    s = s.replace('sqrtf(', 'sqrt(')
    s = s.replace('cosf(', 'cos(')
    s = s.replace('sinf(', 'sin(')
    s = s.replace('FLT_EPSILON', 'DBL_EPSILON')
    return re.sub('([0-9]+\.[0-9]*)f', '\\1', s)

class Values(optparse.Values):
    def __init__(self, *args):
        optparse.Values.__init__(self, *args)
        self.specified = set()

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if hasattr(self, 'specified'):
            self.specified.add(name)

def _parse_range(option, opt_str, value, parser):
    vals = value.split(':')

    # Use higher precision (float64) below.   If single precision is requested, the values
    # will be automatically degraded to float32 later on.
    if len(vals) == 1:
        setattr(parser.values, option.dest, numpy.float64(value))
        parser.par_single.add(option.dest)
    elif len(vals) == 3:
        start = float(vals[0])
        stop = float(vals[1])
        size = float(vals[2])
        setattr(parser.values, option.dest, numpy.linspace(start, stop, size, numpy.float64))
        parser.par_multi.add(option.dest)
    elif len(vals) == 4 and vals[0] == 'log':
        start = float(vals[1])
        stop = float(vals[2])
        size = float(vals[3])
        setattr(parser.values, option.dest, numpy.logspace(start, stop, size, numpy.float64))
        parser.par_multi.add(option.dest)
    else:
        raise OptionValueError('"%s" has to be a single value or a range of the form \'start:stop:steps\'' % opt_str)

def _get_module_source(*files):
    """Load multiple .cu files and join them into a string."""
    src = ""
    for fname in files:
        fp = open(fname)
        src += fp.read()
        fp.close()
    return src

class StateError(Exception):
    pass

class SolverGenerator(object):
    @classmethod
    def get_source(cls, sde, const_parameters, kernel_parameters, local_vars):
        """
        sde: a SDE object for which the code is to be generated
        const_parameters: list of constant parameters for the SDE
        kernel_parameters: list of parameters which will be passed as arguments to
            the kernel
        """
        ctx = {}
        ctx['const_parameters'] = const_parameters
        ctx['local_vars'] = dict([(k, KernelCodePrinter().doprint(v)) for k, v in local_vars.iteritems()])
        ctx['par_cuda'] = kernel_parameters
        ctx['rhs_vars'] = sde.num_vars
        ctx['noise_strength_map'] = sde.noise_map
        ctx['num_noises'] = sde.num_noises
        ctx['sde_code'] = sde.code
        ctx['rng_state_size'] = RNG_STATE[sde.options.rng]
        ctx['rng'] = sde.options.rng

        ctx = cls.update_ctx(sde, ctx)

        lookup = TemplateLookup(directories=sys.path,
                module_directory='/tmp/pysde_modules-%s' %
                                (pwd.getpwuid(os.getuid())[0]))
        sde_template = lookup.get_template('sde.mako')
        return sde_template.render(**ctx)

    @classmethod
    def update_ctx(cls, sde, ctx):
        return ctx


class SRK2(SolverGenerator):
    """Stochastic Runge Kutta method of the 2nd order."""

    @classmethod
    def update_ctx(cls, sde, ctx):
        ctx['method'] = 'SRK2'
        return ctx

class Euler(SolverGenerator):
    @classmethod
    def update_ctx(cls, sde, ctx):
        ctx['method'] = 'Euler'
        return ctx

class Milstein(SolverGenerator):
    @classmethod
    def update_ctx(cls, sde, ctx):
        ctx['method'] = 'Milstein'
        return ctx

class KernelCodePrinter(CCodePrinter):
    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp is basic.S.NegativeOne:
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        # For the kernel code, it's better to calculate the power
        # here explicitly by multiplication.
        elif expr.exp == 2:
            return '%s*%s' % (self.parenthesize(expr.base, PREC),
                              self.parenthesize(expr.base, PREC))
        else:
            return 'powf(%s,%s)' % (self.parenthesize(expr.base, PREC),
                                    self.parenthesize(expr.exp, PREC))

    def _print_Function(self, expr):
        if expr.func.__name__ == 'log':
            return 'logf(%s)' % self.stringify(expr.args, ', ')
        else:
            return super(KernelCodePrinter, self)._print_Function(expr)

class S(object):
    def __init__(self):
        self.dt = Symbol('dt')
        self.samples = Symbol('samples')

class SDE(object):
    """A class representing a SDE equation to solve."""

    format_cmd = r"indent -linux -sob -l120 {file} ; sed -i -e '/^$/{{N; s/\n\([\t ]*}}\)$/\1/}}' -e '/{{$/{{N; s/{{\n$/{{/}}' {file}"

    def __init__(self, code, params, num_vars, num_noises,
            noise_map, period_map=None, args=None, local_vars=None,
            const_pars={}, defaults=None):
        """
        :param code: the code defining the Stochastic Differential Equation
        :param params: dict of simulation parameters; keys are parameter names,
            values are descriptions
        :param num_vars: number of variables in the SDE
        :param num_noises: number of independent, white Gaussian noises
        :param noise_map: a dictionary, mapping the variable number to a list of
            ``num_noises`` noise strengths, which can be either 0 or the name of
            a variable containing the noise strength value
        :param period_map: a dictionary, mapping the variable number to PeriodInfo
            objects.  If a variable has a corresponding entry in this dictionary,
            it will be assumed to be a periodic variable, such that only its
            value modulo ``period`` is important for the evolution of the system.

            Every ``frequency`` * ``samples`` steps, the value
            of this variable will be folded back to the range of [0, period).
            Its full (unfolded) value will however be retained in the output.

            The folded values will result in faster CUDA code if trigonometric
            functions are used and if the magnitude of their arguments always
            remains below 48039.0f (see CUDA documentation).
        :param local_vars: dictionary mapping variable names to functions
            returning SymPy expressions.  The functions should take exactly one
            argument: an instance of the SDE class which called them.  This is
            meant to be used for expressions that depend on the simulation
            parameters.
        :param const_pars: dictionary mapping variable names to functions
            returning NumPy vectors of `sde.num_threads` length.  The functions
            should take exactly one argument: an instance of the SDE class which
            called them.
        :param defaults: dictionary mapping option names to their default
            values.  Any values specified on the command line will override
            default settings specified this way.
        """
        self.parser = OptionParser()

        group = OptionGroup(self.parser, 'SDE engine settings')
        group.add_option('--spp', dest='spp', help='steps per period', metavar='DT', type='int', action='store', default=100)
        group.add_option('--samples', dest='samples', help='sample the position every N steps', metavar='N', type='int', action='store', default=100)
        group.add_option('--paths', dest='paths', help='number of paths to sample', type='int', action='store', default=256)
        group.add_option('--transients', dest='transients', help='number of periods to ignore because of transients', type='int', action='store', default=200)
        group.add_option('--simperiods', dest='simperiods', help='number of periods in the simulation', type='int', action='store', default=2000)
        group.add_option('--seed', dest='seed', help='RNG seed', type='int', action='store', default=None)
        group.add_option('--precision', dest='precision', help='precision of the floating-point numbers (single, double)', type='choice', choices=['single', 'double'], default='single')
        group.add_option('--rng', dest='rng', help='PRNG to use', type='choice', choices=RNG_STATE.keys(), default='kiss32')
        group.add_option('--no-fast-math', dest='fast_math',
                help='do not use faster intrinsic mathematical functions everywhere',
                action='store_false', default=True)
        group.add_option('--deterministic', dest='deterministic', help='do not generate any noises',
                action='store_true', default=False)
        group.add_option('--debug', dest='debug', help='generate debug messages', action='store_true', default=False)
        self.parser.add_option_group(group)

        group = OptionGroup(self.parser, 'Debug settings')
        group.add_option('--save_src', dest='save_src', help='save the generated source to FILE', metavar='FILE',
                               type='string', action='store', default=None)
        group.add_option('--use_src', dest='use_src', help='use FILE instead of the automatically generated code',
                metavar='FILE', type='string', action='store', default=None)
        group.add_option('--noutput_format_src', dest='format_src', help='do not format the generated source code', action='store_false', default=True)
        self.parser.add_option_group(group)

        group = OptionGroup(self.parser, 'Output settings')
        group.add_option('--output_mode', dest='output_mode', help='output mode', type='choice', choices=['summary', 'path'], action='store', default='summary')
        group.add_option('--output_format', dest='output_format', help='output file format', type='choice',
                choices=['npy', 'text', 'store'], action='store', default='npy')
        group.add_option('--output', dest='output', help='base output filename', type='string', action='store', default=None)
        group.add_option('--save_every', dest='save_every', help='save output every N seconds', type='int',
                action='store', default=0)
        self.parser.add_option_group(group)

        group = OptionGroup(self.parser, 'Checkpointing settings')
        group.add_option('--dump_every', dest='dump_every', help='dump system state every N seconds', type='int',
                action='store', default=0)
        group.add_option('--dump_state', dest='dump_filename', help='save state of the simulation to FILE after it is completed',
                metavar='FILE', type='string', action='store', default=None)
        group.add_option('--restore_state', dest='restore_filename', help='restore state of the solver from FILE',
                metavar='FILE', type='string', action='store', default=None)
        group.add_option('--resume', dest='resume', help='resume simulation from a saved checkpoint',
                action='store_true', default=False)
        group.add_option('--continue', dest='continue_', help='continue a finished simulation',
                action='store_true', default=False)
        self.parser.add_option_group(group)

        self.parser.par_multi = set()
        self.parser.par_single = set()

        self.sim_params = params
        self.num_vars = num_vars
        self.num_noises = num_noises
        self.noise_map = noise_map
        self.ext_pars_gen = const_pars

        if period_map is None:
            self.period_map = {}
        else:
            self.period_map = period_map

        self._make_symbols(local_vars)

        # Local variables are defined as lambdas since they need access to the
        # symbols.  Replace the lambdas with their values here now that the
        # symbols are defined.
        self.local_vars = {}
        if local_vars is not None:
            for k, v in local_vars.iteritems():
                self.local_vars[k] = v(self)

        self.code = code

        self.last_dump = time.time()
        self.last_save = time.time()

        # Current results.
        self.state_results = []
        # Results loaded from a dump file.
        self.prev_state_results = []

        self._sim_sym = {}
        self._gpu_sym = {}

        sim_group = OptionGroup(self.parser, 'Simulation-specific settings')

        for name, help_string in params.iteritems():
            sim_group.add_option('--%s' % name, dest=name, action='callback',
                    callback=_parse_range, type='string', help=help_string,
                    default=None)

        self.parser.add_option_group(sim_group)

        for k, v in noise_map.iteritems():
            if len(v) != num_noises:
                raise ValueError('The number of noise strengths for variable %s'
                    ' has to be equal to %d.' % (k, num_noises))

        # Indicates whether simulate() has been run.
        self._simulate_done = False
        # Indicates whether prepare() has been run.
        self._prepared = False
        # Indicates whether the simulation can be continued.
        self._continuable = True

        self.parse_args(args, defaults, sim_group)

        if self.options.deterministic:
            self.num_noises = 0
            self.noise_map = []

    def __del__(self):
        if self._continuable:
            self.finish()

    def _dprint(self, msg):
        if self.options.debug:
            print "#", msg

    def _make_symbols(self, local_vars):
        """Create a sympy Symbol for each simulation parameter."""
        self.S = S()
        for param in self.sim_params.iterkeys():
            setattr(self.S, param, Symbol(param))
        for param in local_vars.iterkeys():
            setattr(self.S, param, Symbol(param))
        for i in range(0, self.num_vars):
            setattr(self.S, 'x%d' % i, Symbol('x%d' % i))
        self.S.t = Symbol('t')

    def parse_args(self, args=None, defaults=None, sim_group=None):
        """Parse command line arguments.

        :param args: string to use instead of sys.argv
        :param defaults: dictionary of default values (option name -> value)
        :param sim_group: OptionGroup containing simulation parameters
        """
        if args is None:
            args = sys.argv

        self.options = Values(self.parser.defaults)
        self.parser.parse_args(args, self.options)

        if defaults is not None:
            for k, v in defaults.iteritems():
                if k not in self.options.specified:
                    # Simulation options need to be processed separated as they
                    # might contain parameter ranges, which need to processed
                    # by _parse_range.
                    if k in self.sim_params and sim_group is not None:
                        for option in sim_group.option_list:
                            if option.dest == k:
                                _parse_range(option, '--%s' % k, str(v), self.parser)
                                break
                    else:
                        setattr(self.options, k, v)

        for name, hs in self.sim_params.iteritems():
            if self.options.__dict__[name] == None:
                raise OptionValueError('Required option "%s" not specified.' % name)

        if self.options.output is None and self.options.output_format != 'text':
            raise OptionValueError('Required option "output"" not specified.')

        if self.options.precision == 'single':
            self.float = numpy.float32
        else:
            self.float = numpy.float64

        if (self.options.resume or self.options.continue_) and self.options.restore_filename is None:
            raise OptionValueError('The resume and continue modes require '
                'a state file to be specified with --restore_state.')

        if self.options.restore_filename is not None:
            self.load_state()

    def _find_cuda_scan_par(self):
        """Automatically determine which parameter is to be scanned over
        in a single kernel run.
        """
        max_par = None
        max = 0

        # Look for a parameter with the highest number of values.
        for par in self.parser.par_multi:
            if len(self.options.__dict__[par]) > max and par != self.freq_var:
                max_par = par
                max = len(self.options.__dict__[par])

        # No variable to scan over.
        if max_par is None:
            return None

        # Remove the selected parameter from the set of multi-valued parameters,
        # over which a host-side loop will be performed when the simulation is
        # executed.
        self.parser.par_multi.remove(max_par)
        return max_par

    def prepare(self, algorithm, init_vectors, freq_var=None):
        """Prepare a SDE simulation.

        :param algorithm: the SDE solver to use, sublass of SDESolver
        :param init_vectors: a callable that will be used to set the initial conditions
        :param freq_var: name of the parameter which is to be interpreted as a
            frequency (determines the step size 'dt').  If ``None``, the
            system period will be assumed to be 1.0 and the time step size
            will be set to 1.0/spp.
        """
        self.freq_var = freq_var

        # Determine the scan variable(s).
        if not (self.options.resume or self.options.continue_):
            sv = self._find_cuda_scan_par()
            if sv is None:
                self.scan_vars = []
            else:
                self.scan_vars = [sv]
                self._dprint("Scanning over '%s' on the GPU." % sv)

            # Create an ordered copy of the list of parameters which have multiple
            # values.
            self.par_multi_ordered = list(self.parser.par_multi)

            # Multi-valued parameters that are scanned over are no longer
            # contained in par_multi at this point.
            self.global_vars = self.parser.par_single | self.parser.par_multi
            userdef_global_vars = self.global_vars.copy()
            self.global_vars.add('dt')
            self.global_vars.add('samples')
            self.const_local_vars = {}

            # If a local variable only depends on constant variables, make
            # it a global constant.
            for name, value in self.local_vars.iteritems():
                if set([str(x) for x in value.atoms(Symbol)]) <= self.global_vars:
                    self.global_vars.add(name)
                    userdef_global_vars.add(name)
                    self.const_local_vars[name] = value

            for name in self.const_local_vars.iterkeys():
                del self.local_vars[name]

        # Prepare the CUDA source.  Load/save it from/to a file if requested.
        if self.options.use_src:
            with open(self.options.use_src, 'r') as file:
                kernel_source = file.read()
        else:
            kernel_source = algorithm.get_source(self, userdef_global_vars,
                   self.scan_vars + list(sorted(self.ext_pars_gen.keys())), self.local_vars)

            if self.options.precision == 'double':
                kernel_source = _convert_to_double(kernel_source)

        if self.options.save_src is not None:
            with open(self.options.save_src, 'w') as file:
                print >>file, kernel_source

            if self.options.format_src:
                os.system(self.format_cmd.format(file=self.options.save_src))

        # Initialize signal handlers in case a dumpfile has been requested.
        if self.options.dump_filename is not None:
            signal.signal(signal.SIGUSR1, _sighandler)
            signal.signal(signal.SIGINT, _sighandler)
            signal.signal(signal.SIGQUIT, _sighandler)
            signal.signal(signal.SIGHUP, _sighandler)
            signal.signal(signal.SIGTERM, _sighandler)

        signal.signal(signal.SIGUSR2, _sighandler)
        self._prepared = True

        return self.cuda_prep(init_vectors, kernel_source)

    @property
    def scan_var_size(self):
        ret = 1

        for par in self.scan_vars:
            ret *= len(getattr(self.options, par))

        return ret

    def get_param(self, name):
        """Get the value of a simulation parameter.

        name: name of the parameter to get
        """
        if name in self.scan_vars:
            idx = self.scan_vars.index(name)
            return self._sv[idx]
        elif name in self._sim_sym:
            return self._sim_sym[name]
        else:
            return getattr(self.options, name)

    def set_param(self, name, value):
        """Set the value of a simulation parameter.

        Note that multiply-valued parameters (i.e. parameters over which a
        scan is being perfomed), cannot be assigned to in this way.

        :param name: name of the parameter to set
        :param value: value of the parameter to set
        """
        # Multi-valued paramers cannot be programmatically reset.
        assert name not in self.scan_vars
        assert name in self._sim_sym

        self._sim_sym[name] = value

    def cuda_prep(self, init_vectors, sources, sim_func='AdvanceSim'):
        """Prepare a SDE simulation for execution using CUDA.

        :param init_vectors: a function which takes an instance of this class and the
            variable number as arguments and returns an initialized vector of
            size num_threads
        :param sources: list of source code files for the simulation
        :param sim_func: name of the kernel advacing the simulation in time
        """
        if self.options.seed is not None and not (self.options.resume or self.options.continue_):
            numpy.random.seed(self.options.seed)

        self.init_vectors = init_vectors
        self.num_threads = self.scan_var_size * self.options.paths
        self._sim_prep_mod(sources, sim_func)
        self._sim_prep_const()
        self._sim_prep_var()

    def _sim_prep_mod(self, sources, sim_func):
        # The use of fast math below will result in certain mathematical functions
        # being automaticallky replaced with their faster counterparts prefixed with
        # __, e.g. __sinf().
        if self.options.fast_math:
            options=['--use_fast_math']
        else:
            options=[]

        if type(sources) is str:
            self.mod = pycuda.compiler.SourceModule(sources, options=options)
        else:
            self.mod = pycuda.compiler.SourceModule(_get_module_source(*sources), options=options)
        self.advance_sim = self.mod.get_function(sim_func)

    def _sim_prep_const(self):
        for var in self.global_vars:
            self._gpu_sym[var] = self.mod.get_global(var)[0]

        samples = numpy.uint32(self.options.samples)
        cuda.memcpy_htod(self._gpu_sym['samples'], samples)

        # Single-valued system parameters
        for par in self.parser.par_single:
            self._sim_sym[par] = self.options.__dict__[par]
            cuda.memcpy_htod(self._gpu_sym[par], self.float(self.options.__dict__[par]))

    def _init_rng(self):
        # Initialize the RNG seeds.
        self._dprint('Initializing RNG seeds.')
        self._rng_state = numpy.fromstring(numpy.random.bytes(
            self.num_threads * RNG_STATE[self.options.rng] * numpy.uint32().nbytes),
            dtype=numpy.uint32)
        self._gpu_rng_state = cuda.mem_alloc(self._rng_state.nbytes)
        cuda.memcpy_htod(self._gpu_rng_state, self._rng_state)

    def _sim_prep_var(self):
        self.vec = []
        self.ext_pars = []
        self._gpu_ext_pars = []
        self._gpu_vec = []

        # Prepare device vectors.
        for i in range(0, self.num_vars):
            vt = numpy.zeros(self.num_threads).astype(self.float)
            self.vec.append(vt)
            self._gpu_vec.append(cuda.mem_alloc(vt.nbytes))

        for par_name, par_gen in sorted(self.ext_pars_gen.iteritems()):
            vt = numpy.zeros(self.num_threads).astype(self.float)
            self.ext_pars.append(vt)
            self._gpu_ext_pars.append(cuda.mem_alloc(vt.nbytes))

        self._sv = []
        self._gpu_sv = []

        # Initialize the scan variables.  If we're in resume or continue mode, the
        # scan variables are already initialized and nothing needs to be done.
        if not self.options.resume and not self.options.continue_:
            for sv in self.scan_vars:
                vals = numpy.ones(self.options.paths)
                for sv2 in reversed(self.scan_vars):
                    if sv2 != sv:
                        vals = numpy.kron(numpy.ones_like(getattr(self.options, sv2)), vals)
                    else:
                        vals = numpy.kron(getattr(self.options, sv2), vals)

                self._sv.append(vals.astype(self.float))

        for sv in self._sv:
            self._gpu_sv.append(cuda.mem_alloc(sv.nbytes))
            cuda.memcpy_htod(self._gpu_sv[-1], sv)



    def get_var(self, i, starting=False):
        """Return a vector of the i-th dynamical variable.

        :param i: the dynamical variable to be returned
        :param starting: if True and if running in summary more, makes this method
            return the value of the chosen dynamical variable as measured at
            the reference time point (after `transients` periods).  Otherwise,
            the current value is returned.
        """
        if starting:
            vec = self.vec_start
            nx = self.vec_start_nx
        else:
            vec = self.vec
            nx = self.vec_nx

        # Unfold the variable if necessary.
        if i in nx:
            return vec[i] + self.period_map[i].period * nx[i]
        else:
            return vec[i]

    @property
    def max_sim_iter(self):
        return int(self.options.simperiods * self.options.spp / self.options.samples)+1

    def iter_to_sim_time(self, iter_):
        return iter_ * self.dt * self.options.samples

    def sim_time_to_iter(self, time_):
        return int(time_ / (self.dt * self.options.samples))

    def simulate(self, req_output, block_size=64, continuable=False):
        """Run a CUDA SDE simulation.

        :param req_output: a dictionary mapping the the output mode to a list of
            tuples of ``(callable, vars)``, where ``callable`` is a function
            that will compute the values to be returned, and ``vars`` is a list
            of variables that will be passed to this function
        :param block_size: CUDA block size
        :param continuable: if True, the simulation can be continued (possibly after
            changing some of the parameters) using :meth:`continue_simulation`
        """
        if not self._prepared:
            raise StateError('Call the prepare() method before running a simulation')

        # Initialize the output module.
        self.req_output = req_output[self.options.output_mode]

        if self.options.output_format == 'text':
            self.output = output.TextOutput(self, self.req_output.keys())
        elif self.options.output_format == 'npy':
            self.output = output.NpyOutput(self, self.req_output.keys())
        elif self.options.output_format == 'store':
            self.output = output.StoreOutput(self, self.req_output.keys())

        self.output.header()

        # Determine which variables are necessary for the output.
        self.req_vars = set([])
        for k, v in self.req_output.iteritems():
            for func, vars in v:
                self.req_vars |= set(vars)

        # Initialize the CUDA kernel.
        self.block_size = block_size
        arg_types = ['P'] + ['P']*self.num_vars + ['P'] * (len(self.scan_vars) +
                len(self.ext_pars_gen.keys())) + [self.float]
        self.advance_sim.prepare(arg_types, block=(block_size, 1, 1))

        # Print stats about the kernel.
        kern = self.advance_sim
        ddata = pycuda.tools.DeviceData()
        occ = pycuda.tools.OccupancyRecord(ddata, block_size,
                kern.shared_size_bytes, kern.num_regs)

        print '# CUDA stats l:%d  s:%d  r:%d  occ:(%f tb:%d w:%d l:%s)' % (
                kern.local_size_bytes, kern.shared_size_bytes,
                kern.num_regs, occ.occupancy, occ.tb_per_mp,
                occ.warps_per_mp, occ.limited_by)

        self._scan_iter = 0
        # Set to True so that continue_simulation will always run.  If the
        # simulation is not continuable, this variable will be set to False in
        # continue_simulation.
        self._continuable = True
        self.continue_simulation(continuable)
        self._simulate_done = True

    def continue_simulation(self, continuable=False):
        if not self._continuable:
            raise StateError("This simulation cannot be continued.")

        self._run_nested(self.par_multi_ordered)
        self._continuable = continuable

        if not continuable:
            self.finish()

        if self.options.dump_filename is not None:
            self.dump_state()

    def finish(self):
        # If simulate() has not been run yet, the output object is not defined.
        if self._simulate_done:
            self.output.close()
        self._continuable = False

    def _run_nested(self, range_pars):
        """Process parameters with multiple values and start a SDE simulation.

        :param range_pars: iterable of names of parameters that have more than one
            value
        """
        # No more parameters to loop over, time to actually run the kernel.
        if not range_pars:
            # Reinitialize the RNG here so that there is no interdependence
            # between runs.  This also guarantees that the resume/continue
            # modes can work correctly in the case of a scan over 2+ parameters.
            self._init_rng()

            # Calculate period and step size.  The frequency variable is
            # guaranteed to be constant during a single kernel run, i.e. the
            # scan over frequency is performed via a loop on the host.
            if self.freq_var is not None:
                period = 2.0 * math.pi / self._sim_sym[self.freq_var]
            else:
                period = 1.0
            self.dt = self.float(period / self.options.spp)
            cuda.memcpy_htod(self._gpu_sym['dt'], self.dt)
            self._dprint("Setting 'dt' = %f" % self.dt)

            # Evaluate constant local vars.
            subs = {self.S.dt: self.dt}
            for k, v in self._sim_sym.iteritems():
                subs[Symbol(k)] = v

            for name, value in self.const_local_vars.iteritems():
                val = self.float(value.subs(subs))
                self._dprint("Setting '%s' = %f" % (name, val))
                cuda.memcpy_htod(self._gpu_sym[name], val)

            # In the resume mode, we skip all the computations that have already
            # been completed and thus are saved in self.prev_state_results.
            if not self.options.resume:
                self._run_kernel(period)
            elif (self._scan_iter == len(self.prev_state_results)-1 and
                    self.prev_state_results[-1][0] < self.iter_to_sim_time(self.max_sim_iter)):
                self._run_kernel(period)
            elif self._scan_iter > len(self.prev_state_results)-1:
                self._run_kernel(period)

            self._scan_iter += 1
        else:
            par = range_pars[0]

            # Loop over all values of a specific parameter.
            for val in self.options.__dict__[par]:
                self._sim_sym[par] = self.float(val)
                self._dprint("Setting '%s' = %f" % (par, self._sim_sym[par]))
                cuda.memcpy_htod(self._gpu_sym[par], self.float(val))
                self._run_nested(range_pars[1:])

    def _restore_state(self, period):
        self._dprint('Restoring simulation state.')
        self.vec = self.prev_state_results[self._scan_iter][1]
        self.vec_nx = self.prev_state_results[self._scan_iter][2]
        self._rng_state = self.prev_state_results[self._scan_iter][3]
        cuda.memcpy_htod(self._gpu_rng_state, self._rng_state)
        for i in range(0, self.num_vars):
            cuda.memcpy_htod(self._gpu_vec[i], self.vec[i])

        self.sim_t = self.prev_state_results[self._scan_iter][0]

        if self.options.output_mode == 'summary':
            self.start_t = self.prev_state_results[self._scan_iter][4]
            self.vec_start = self.prev_state_results[self._scan_iter][5]
            self.vec_start_nx = self.prev_state_results[self._scan_iter][6]

            if self.sim_t >= self.options.transients * period:
                return False
            return True
        else:
            return False

    def _init_vectors(self, transient):
        # Reinitialize the positions.
        self._dprint('Initializing positions.')
        self.vec = []
        for i in range(0, self.num_vars):
            vt = self.init_vectors(self, i).astype(self.float)
            self.vec.append(vt)
            cuda.memcpy_htod(self._gpu_vec[i], vt)

        # Prepare an array for number of periods for periodic variables.
        self.vec_nx = {}
        for i, v in self.period_map.iteritems():
            self.vec_nx[i] = numpy.zeros_like(self.vec[i]).astype(numpy.int64)

        if transient:
            self.vec_start = []
            for i in range(0, self.num_vars):
                self.vec_start.append(numpy.zeros_like(self.vec[i]))

    def _init_ext_pars(self):
        self.ext_pars = []
        # Generate the values of the external parameters.
        for i, (par_name, par_gen) in enumerate(sorted(self.ext_pars_gen.iteritems())):
            vt = par_gen(self).astype(self.float)
            self.ext_pars.append(vt)
            self._dprint("Setting '%s' = %f" % (par_name, vt))
            cuda.memcpy_htod(self._gpu_ext_pars[i], vt)

    def _run_kernel(self, period):
        kernel_args = [self._gpu_rng_state] + self._gpu_vec + self._gpu_sv + self._gpu_ext_pars
        global want_dump, want_exit, want_save

        if self.options.output_mode == 'path':
            transient = False
            pathwise = True
        else:
            transient = True
            pathwise = False

        if (self.options.continue_ or
                (self.options.resume and self._scan_iter < len(self.prev_state_results))):
            transient = self._restore_state(period)
            # FIXME: Support external paremeters in resumed simulations.
        else:
            # If this is a continued simulation, there is no need to
            # reinitialize the vectors.
            if not self._simulate_done:
                self._init_vectors(transient)
            self._init_ext_pars()
            self.sim_t = 0.0

        init_iter = self.sim_time_to_iter(self.sim_t)

        def fold_variables(iter_, need_copy):
            """Reduce periodic variables to the base domain.

            :param need_copy: if True, the variable needs to be fetched from the
                GPU
            """
            for i, (period, freq) in self.period_map.iteritems():
                if iter_ % freq == 0:
                    if need_copy:
                        cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])

                    self.vec_nx[i] = numpy.add(self.vec_nx[i],
                            numpy.floor_divide(self.vec[i], period).astype(numpy.int64))
                    self.vec[i] = numpy.remainder(self.vec[i], period)
                    cuda.memcpy_htod(self._gpu_vec[i], self.vec[i])

        # TODO: if transients are requested, split the kernel call into
        # two parts -- before the transient, and after.
        # Actually run the simulation here.
        for j in xrange(init_iter, self.max_sim_iter):
            self.sim_t = self.iter_to_sim_time(j)

            if transient and self.sim_t >= self.options.transients * period:
                for i in range(0, self.num_vars):
                    cuda.memcpy_dtoh(self.vec_start[i], self._gpu_vec[i])
                transient = False
                self.start_t = self.sim_t
                self.vec_start_nx = self.vec_nx.copy()

            # Fold the time variable passed to the kernel.
            # NOTE: Here we implicitly assume that only the reduced value of
            # time matters for the evolution of the system.
            args = kernel_args + [self.float(math.fmod(self.sim_t, period))]
            self.advance_sim.prepared_call((self.num_threads/self.block_size, 1), *args)
            self.sim_t += self.options.samples * self.dt

            # See whether a data save was requested.
            if (self.options.save_every > 0 and
                    (time.time() - self.last_save > self.options.save_every)):
                want_save = True
                self.last_save = time.time()

            fold_variables(j, True)

            # Save the current data if running in pathwise mode.
            if pathwise:
                self.output_current()
                if self.scan_vars:
                    self.output.finish_block()
                    if want_save:
                        self.output.flush()
                        want_save = False

            # Create a dump file if requested.
            if (want_dump or
                    (self.options.dump_every > 0 and
                     time.time() - self.last_dump > self.options.dump_every)):

                if self.options.dump_filename is not None:
                    self.save_block()
                    self.dump_state()
                    del self.state_results[-1]
                want_dump = False

                if want_exit:
                    sys.exit(0)

        # Only calculate the summary values if we are running in summary mode.
        if not pathwise:
            self.output_summary()

        self.output.finish_block()
        if want_save:
            self.output.flush()
            want_save = False
        self.save_block()

    def output_current(self):
        """Pass the results of the current iteration to the output module."""
        dyn_vars = {}
        # Get required variables from the compute device and unfold them.
        for i in self.req_vars:
            cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])
            dyn_vars[i] = self.get_var(i)

        self._output_results(dyn_vars, self.sim_t)

    def output_summary(self):
        dyn_vars = {}
        # Get required variables from the compute device and unfold them.
        for i in self.req_vars:
            cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])
            # For each variable, store both the reference (starting) value
            # and the final one.
            dyn_vars[i] = (self.get_var(i, starting=True), self.get_var(i))

        self._output_results(dyn_vars)

    def _output_results(self, vars, *misc_pars):
        """Prepare data for the output module."""
        # Iterate over all values of the scan parameters.  For each unique
        # value, calculate the requested output(s).
        for i in range(0, self.scan_var_size):
            out = {'main': list(misc_pars)}

            for out_name, out_decl in self.req_output.iteritems():
                for decl in out_decl:
                    # Map variable numbers to their actual values.
                    args = map(lambda x: vars[x], decl.vars)

                    # Cut the part of the variables for the current value of the scan vars.
                    if args and type(args[0]) is tuple:
                        args = map(lambda x:
                            (x[0][i*self.options.paths:(i+1)*self.options.paths],
                             x[1][i*self.options.paths:(i+1)*self.options.paths]), args)
                    else:
                        args = map(lambda x:
                                x[i*self.options.paths:(i+1)*self.options.paths],
                                args)

                    # Evaluate the requested function.
                    out.setdefault(out_name, []).extend(decl.func(self, *args))

            self.output.data(**out)

    @property
    def state(self):
        """A dictionary representing the current state of the solver."""

        names = ['sim_params', 'num_vars', 'num_noises',
                 'noise_map', 'period_map', 'code', 'options', 'float',
                 'scan_vars', 'local_vars', 'const_local_vars', 'global_vars',
                 'par_multi_ordered']

        ret = {}

        for name in names:
            ret[name] = getattr(self, name)

        ret['par_single'] = self.parser.par_single
        ret['par_multi'] = self.parser.par_multi

        return ret

    def save_block(self):
        """Save the current block into the state of the solver if necessary."""

        # If no dump file is specified, do not store any data in state_results.
        if self.options.dump_filename is None:
            return

        cuda.memcpy_dtoh(self._rng_state, self._gpu_rng_state)
        for i in range(0, self.num_vars):
            cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])

        if self.options.output_mode == 'path':
            self.state_results.append((self.sim_t, copy.deepcopy(self.vec), self.vec_nx.copy(),
                self._rng_state.copy()))
        else:
             self.state_results.append((self.sim_t, copy.deepcopy(self.vec), self.vec_nx.copy(),
                self._rng_state.copy(), self.start_t, copy.deepcopy(self.vec_start),
                self.vec_start_nx.copy()))

    def dump_state(self):
        """Dump the current state of the solver to a file.

        This makes it possible to later restart the calculations from the saved
        checkpoint using the :meth:`load_state` function.
        """
        state = self.state
        state['results'] = self.state_results
        state['numpy.random'] = numpy.random.get_state()

        if self.scan_vars:
            state['sv'] = copy.deepcopy(self._sv)

        with open(self.options.dump_filename, 'w') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

        self.last_dump = time.time()

    def load_state(self):
        """Restore saved state of the solver.

        After the state is restored, the simulation can be continued the standard
        way, i.e. by calling :meth:`prepare` and :meth:`simulate`.
        """
        with open(self.options.restore_filename, 'r') as f:
            state = pickle.load(f)

        # The current options object will be overriden by the one saved in the
        # checkpoint file.
        new_options = self.options
        for par, val in state.iteritems():
            if par not in ['par_single', 'par_multi', 'results', 'numpy.random', 'sv']:
                setattr(self, par, val)

        self.parser.par_single = state['par_single']
        self.parser.par_multi = state['par_multi']
        self.prev_state_results = state['results']

        numpy.random.set_state(state['numpy.random'])

        if 'sv' in state:
            self._sv = state['sv']

        # Options overridable from the command line.
        overridable = ['resume', 'continue_', 'dump_filename']

        # If this is a continuation of a previous simulation, make output-related
        # parameters overridable.
        if new_options.continue_:
            overridable.extend(['output', 'output_format', 'output_mode', 'simperiods'])
            # TODO: This could potentially cause problems with transients if the original
            # simulation was run in summary mode and the new one is in path mode.
        else:
            self.state_results = self.prev_state_results

        for option in overridable:
            if hasattr(new_options, option) and getattr(new_options, option) is not None:
                setattr(self.options, option, getattr(new_options, option))

