import math
import os
import pwd
import sys

from optparse import OptionParser, OptionValueError, Values

import numpy

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler

from mako.template import Template
from mako.lookup import TemplateLookup

# Map RNG name to number of uint state variables.
RNG_STATE = {
    'xs32': 1,
    'kiss32': 4,
    'nr32': 4,
}

def drift_velocity(sde, *args):
    ret = []
    for starting, final in args:
        a = starting.astype(numpy.float64)
        b = final.astype(numpy.float64)

        ret.append((numpy.average(b) - numpy.average(a)) /
                (sde.sim_t - sde.start_t))

    return ret

def diffusion_coefficient(sde, *args):
    ret = []
    for starting, final in args:
        a = starting.astype(numpy.float64)
        b = final.astype(numpy.float64)

        deff1 = numpy.average(numpy.square(b)) - numpy.average(b)**2
        deff2 = numpy.average(numpy.square(a)) - numpy.average(a)**2
#        ret.append((deff1 - deff2) / (2.0 * (sde.sim_t - sde.start_t)))
        ret.append(deff1 / (2.0 * sde.sim_t))
        ret.append(deff2 / (2.0 * sde.start_t))

    return ret

def avg_moments(sde, *args):
    ret = []

    for arg in args:
        ret.append(numpy.average(arg))
        ret.append(numpy.average(numpy.square(arg)))

    return ret

def _convert_to_double(src):
    import re
    s = src.replace('float', 'double')
    s = s.replace('FLT_EPSILON', 'DBL_EPSILON')
    return re.sub('([0-9]+\.[0-9]*)f', '\\1', s)

def _parse_range(option, opt_str, value, parser):
    vals = value.split(':')

    # Use higher precision (float64) below.   If single precision is requested, the values
    # will be automatically degraded to float32 later on.
    if len(vals) == 1:
        setattr(parser.values, option.dest, numpy.float64(value))
        parser.par_single.append(option.dest)
    elif len(vals) == 3:
        start = float(vals[0])
        stop = float(vals[1])
        step = (stop-start) / (int(vals[2])-1)
        setattr(parser.values, option.dest, numpy.arange(start, stop+0.999*step, step, numpy.float64))
        parser.par_multi.append(option.dest)
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


class SolverGenerator(object):
    @classmethod
    def get_source(cls, sde, parameters, kernel_parameters):
        """
        sde: a SDE object for which the code is to be generated
        parameters: list of parameters for the SDE
        kernel_parameters: list of parameters which will be passed as arguments to
            the kernel
        """
        pass

class SRK2(SolverGenerator):
    """Stochastic Runge Kutta method of the 2nd order."""

    @classmethod
    def get_source(cls, sde, parameters, kernel_parameters):
        noise_strengths = set()
        for i in sde.noise_map.itervalues():
            noise_strengths.update(set(i))
        noise_strengths.difference_update(set([0]))

        num_noises = sde.num_noises + sde.num_noises % 2

        ctx = {}
        ctx['const_parameters'] = parameters - set(kernel_parameters) | set(noise_strengths)
        ctx['par_cuda'] = kernel_parameters
        ctx['rhs_vars'] = sde.num_vars
        ctx['noise_strength_map'] = sde.noise_map
        ctx['noises'] = sde.num_noises
        ctx['num_noises'] = num_noises
        ctx['sde_code'] = sde.code
        ctx['rng_state_size'] = RNG_STATE[sde.options.rng]
        ctx['rng'] = sde.options.rng

        lookup = TemplateLookup(directories=sys.path,
                module_directory='/tmp/pysde_modules-%s' %
                                (pwd.getpwuid(os.getuid())[0]))
        sde_template = lookup.get_template('sde.mako')
        return sde_template.render(**ctx)


class TextOutput(object):
    def __init__(self, sde):
        self.sde = sde

    def finish_block(self):
        print

    def data(self, pars):
        rep = ['%15.8e' % x for x in pars]
        print ' '.join(rep)

    def header(self):
        print '# %s' % ' '.join(sys.argv)
        if self.sde.options.seed is not None:
            print '# seed = %d' % self.sde.options.seed
        print '# sim periods = %d' % self.sde.options.simperiods
        print '# transient periods = %d' % self.sde.options.transients
        print '# samples = %d' % self.sde.options.samples
        print '# paths = %d' % self.sde.options.paths
        print '# spp = %d' % self.sde.options.spp
        for par in self.sde.parser.par_single:
            print '# %s = %f' % (par, self.sde.options.__dict__[par])
        print '#',
        for par in self.sde.parser.par_multi:
            print par,

        if self.sde.scan_var is not None:
            print '%s' % self.sde.scan_var,
        for i in range(0, self.sde.num_vars):
            print 'x%d' % i,

        print

class LoggerOutput(object):
    def __init__(self, sde):
        self.sde = sde
        self.log = []

    def finish_block(self):
        pass

    def data(self, pars):
        self.log.append(pars)

    def header(self):
        pass

class HDF5Output(object):
    def __init__(self, sde):
        self.sde = sde
        import tables

    def finish_block(self):
        self.h5table.flush()

    def data(self, pars):
        record = self.h5table.row
        for i, col in enumerate(self.h5table.cols._v_colnames):
            record[col] = pars[i]
        record.append()

    def header(self):
        desc = {}
        pars = []
        pars.extend(self.sde.parser.par_multi)

        if self.sde.scan_var is not None:
            pars.append(sde.scan_var)

        if self.sde.options.omode == 'path':
            pars.append('t')
        for i in range(0, self.sde.num_vars):
            pars.append('x%d' % i)

        for i, par in enumerate(pars):
            desc[par] = tables.Float32Col(pos=i)

        self.h5file = tables.openFile('output.h5', mode = 'w')
        self.h5group = self.h5file.createGroup('/', 'results', 'simulation results')
        self.h5table = self.h5file.createTable(self.h5group, 'results', desc, 'results')


class SDE(object):
    """A class representing a SDE equation to solve."""

    format_cmd = r"indent -linux -sob -l120 {file} ; sed -i -e '/^$/{{N; s/\n\([\t ]*}}\)$/\1/}}' -e '/{{$/{{N; s/{{\n$/{{/}}' {file}"

    def __init__(self, code, params, global_vars, num_vars, num_noises,
            noise_map, periodic_map=None):
        """
        :param code: the code defining the Stochastic Differential Equation
        :param params: list of simulation parameters defined as tuples
            (param name, param description)
        :param global_vars: list of global symbols in the CUDA code
        :param num_vars: number of variables in the SDE
        :param num_noises: number of independent, white Gaussian noises
        :param noise_map: a dictionary, mapping the variable number to a list of
            ``num_noises`` noise strengths, which can be either 0 or the name of
            a constant CUDA variable containing the noise strength value
        :param periodic_map: a dictionary, mapping the variable number to tuples
            of (period, frequency).  If a variable has a corresponding entry in
            this dictionary, it will be assumed to be a periodic variable, such
            that only its value modulo ``period`` is important for the evolution
            of the system.  Every ``frequency`` * ``samples`` steps, the value
            of this variable will be folded back to the range of [0, period).
            It's full (unfolded) value will however be retained in the output.

            The folded values will result in faster CUDA code if trigonometric
            functions are used and if the magnitude of their arguments always
            remains below 48039.0f (see CUDA documentation).
        """
        self.parser = OptionParser()
        self.parser.add_option('--spp', dest='spp', help='steps per period', metavar='DT', type='int', action='store', default=100)
        self.parser.add_option('--samples', dest='samples', help='sample the position every N steps', metavar='N', type='int', action='store', default=100)
        self.parser.add_option('--paths', dest='paths', help='number of paths to sample', type='int', action='store', default=256)
        self.parser.add_option('--transients', dest='transients', help='number of periods to ignore because of transients', type='int', action='store', default=200)
        self.parser.add_option('--simperiods', dest='simperiods', help='number of periods in the simulation', type='int', action='store', default=2000)
        self.parser.add_option('--output_mode', dest='omode', help='output mode', type='choice', choices=['summary', 'path'], action='store', default='summary')
        self.parser.add_option('--seed', dest='seed', help='RNG seed', type='int', action='store', default=None)
        self.parser.add_option('--output_format', dest='oformat', help='output file format', type='choice',
                choices=['text', 'hdf_expanded', 'hdf_nested', 'logger'], action='store', default='text')
        self.parser.add_option('--save_src', dest='save_src', help='save the generated source to FILE', metavar='FILE',
                               type='string', action='store', default=None)
        self.parser.add_option('--use_src', dest='use_src', help='use FILE instead of the automatically generated code',
                metavar='FILE', type='string', action='store', default=None)
        self.parser.add_option('--precision', dest='precision', help='precision of the floating-point numbers (single, double)', type='choice', choices=['single', 'double'], default='single')
        self.parser.add_option('--rng', dest='rng', help='PRNG to use', type='choice', choices=RNG_STATE.keys(), default='kiss32')
        self.parser.add_option('--noformat_src', dest='format_src', help='do not format the generated source code', action='store_false', default=True)
        self.parser.add_option('--no-fast-math', dest='fast_math',
                help='do not use faster intrinsic mathematical functions everywhere',
                action='store_false', default=True)

        # List of single-valued system parameters
        self.parser.par_multi = []
        # List of multi-valued system parameters
        self.parser.par_single = []
        self.sim_params = params
        self.global_vars = global_vars
        self.num_vars = num_vars
        self.num_noises = num_noises
        self.noise_map = noise_map
        if periodic_map is None:
            self.periodic_map = {}
        else:
            self.periodic_map = periodic_map
        self.code = code

        self._sim_sym = {}
        self._gpu_sym = {}

        # Additional global symbols which are defined for every simulation.
        global_vars.append('samples');
        global_vars.append('dt');

        # By default, assume that all parameters are constants during a single run.
        # This might not be the case if one of the parameters will be scanned over
        # in the kernel, in which case it will be removed from the list at a later
        # time.
        for par, desc in params:
            global_vars.append(par)

        for name, help_string in params:
            self.parser.add_option('--%s' % name, dest=name, action='callback',
                    callback=_parse_range, type='string', help=help_string,
                    default=None)

        for k, v in noise_map.iteritems():
            if len(v) != num_noises:
                raise ValueError('The number of noise strengths for variable %s'
                    'has to be equal to %d.' % (k, num_noises))

    def parse_args(self, args=None):
        if args is None:
            args = sys.argv

        self.options = Values(self.parser.defaults)
        self.parser.parse_args(args, self.options)

        opt_ok = True
        for name, hs in self.sim_params:
            if self.options.__dict__[name] == None:
                print 'Required option "%s" not specified.' % name
                opt_ok = False

        if self.options.precision == 'single':
            self.float = numpy.float32
        else:
            self.float = numpy.float64

        if self.options.oformat == 'text':
            self.output = TextOutput(self)
        elif self.options.oformat ==  'logger':
            self.output = LoggerOutput(self)
        else:
            self.output = HDF5Output(self)

        return opt_ok

    # TODO: add support for scanning over a set of parameters
    def _find_cuda_scan_par(self):
        """Automatically determine which parameter is to be scanned over
        in a single kernel run.
        """
        max_i = -1
        max_par = None
        max = 0

        # Look for a parameter with the highest number of values.
        for i, par in enumerate(self.parser.par_multi):
            if len(self.options.__dict__[par]) > max:
                max_par = par
                max = len(self.options.__dict__[par])
                max_i = i

        # No variable to scan over.
        if max_i < 0:
            return None

        # Remove this parameter from the list of parameters with multiple
        # values, so that we don't loop over it when running the simulation.
        del self.parser.par_multi[max_i]
        self.global_vars.remove(max_par)

        return max_par

    def prepare(self, algorithm, init_vectors):
        """Prepare a SDE simulation.

        :param algorithm: the SDE solver to use, sublass of SDESolver
        :param init_vectors: a callable that will be used to set the initial conditions
        """
        scan_var = self._find_cuda_scan_par()

        if scan_var is not None:
            scan_set = set([scan_var])
        else:
            scan_set = set([])

        if self.options.use_src:
            with open(self.options.use_src, 'r') as file:
                kernel_source = file.read()
        else:
            kernel_source = algorithm.get_source(self,
                    set(self.global_vars) - set(['dt', 'samples']), scan_set)

            if self.options.precision == 'double':
                kernel_source = _convert_to_double(kernel_source)

        if self.options.save_src is not None:
            with open(self.options.save_src, 'w') as file:
                print >>file, kernel_source

            if self.options.format_src:
                os.system(self.format_cmd.format(file=self.options.save_src))

        return self.cuda_prep(init_vectors, kernel_source, scan_var)

    @property
    def scan_var_size(self):
        if self.scan_var is not None:
            return len(getattr(self.options, self.scan_var))
        else:
            return 1

    @property
    def scan_values(self):
        if self.scan_var is not None:
            return getattr(self.options, self.scan_var)
        else:
            return None

    def cuda_prep(self, init_vectors, sources, scan_var,
                  sim_func='AdvanceSim'):
        """Prepare a SDE simulation for execution using CUDA.

        init_vectors: a function which takes an instance of this class and the
            variable number as arguments and returns an initialized vector of
            size num_threads
        sources: list of source code files for the simulation
        scan_var: name of the scan variable.  The scan variable is a system
            parameter for whose multiple values results are obtained in a
            single CUDA kernel launch)
        sim_func: name of the kernel advacing the simulation in time
        """
        if self.options.seed is not None:
            numpy.random.seed(self.options.seed)

        self.init_vectors = init_vectors
        self.scan_var = scan_var
        self.num_threads = self.scan_var_size * self.options.paths
        self._sim_prep_mod(sources, sim_func)
        self._sim_prep_const()
        self._sim_prep_var()

        self.output.header()

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

        # Simulation parameters
        samples = numpy.uint32(self.options.samples)
        cuda.memcpy_htod(self._gpu_sym['samples'], samples)

        # Single-valued system parameters
        for par in self.parser.par_single:
            self._sim_sym[par] = self.options.__dict__[par]

            # If a variable is not in the dictionary, then it is automatically
            # calculated and will be set at a later stage.
            if par in self._gpu_sym:
                cuda.memcpy_htod(self._gpu_sym[par], self.float(self.options.__dict__[par]))

    def _sim_prep_var(self):
        self.vec = []
        self._gpu_vec = []

        # Prepare device vectors.
        for i in range(0, self.num_vars):
            vt = numpy.zeros(self.num_threads).astype(self.float)
            self.vec.append(vt)
            self._gpu_vec.append(cuda.mem_alloc(vt.nbytes))

        # Initialize the RNG seeds.
        self._rng_state = numpy.random.randint(0, 2**32-1, self.num_threads * RNG_STATE[self.options.rng])
        self._rng_state = self._rng_state.astype(numpy.uint32)
        self._gpu_rng_state = cuda.mem_alloc(self._rng_state.nbytes)
        cuda.memcpy_htod(self._gpu_rng_state, self._rng_state)

        if self.scan_var is None:
            return

        # Initialize the scan variable.
        self._sv = numpy.kron(self.scan_values, numpy.ones(self.options.paths)).astype(self.float)
        self._gpu_sv = cuda.mem_alloc(self._sv.nbytes)
        cuda.memcpy_htod(self._gpu_sv, self._sv)

    def set_param(self, name, val):
        self._sim_sym[name] = val
        cuda.memcpy_htod(self._gpu_sym[name], self.float(val))

    def get_param(self, name):
        try:
            return self._sim_sym[name]
        except KeyError:
            if name in self.scan_var:
                return self._sv
            else:
                return getattr(self.options, name)

    def get_var(self, i, starting=False):
        if starting:
            vec = self.vec_start
            nx = self.vec_start_nx
        else:
            vec = self.vec
            nx = self.vec_nx

        if i in nx:
            return vec[i] + self.periodic_map[i][0] * nx[i]
        else:
            return vec[i]

    def simulate(self, req_output, calculated_params, block_size=64, freq_var=None):
        """Run a CUDA SDE simulation.

        req_output: a dictionary mapping the the output mode to a list of
            tuples of ``(callable, vars)``, where ``callable`` is a function
            that will compute the values to be returned, and ``vars`` is a list
            of variables that will be passed to this function
        calculated_params: a function, which given an instance of this
            class will setup the values of automatically calculated
            parameters
        block_size: CUDA block size
        freq_var: name of the parameter which is to be interpreted as a
            frequency (determines the step size 'dt').  If ``None``, the
            system period will be assumed to be 1.0 and the time step size
            will be set to 1.0/spp.
        """
        self.req_output = req_output[self.options.omode]

        # Determine which variables are necessary for the output.
        self.req_vars = set([])
        for i, v in self.req_output:
            self.req_vars |= set(v)

        self.block_size = block_size
        arg_types = ['P'] + ['P']*self.num_vars

        if self.scan_var is not None:
            arg_types += ['P']

        arg_types += [self.float]
        self.advance_sim.prepare(arg_types, block=(block_size, 1, 1))
        self._run_nested(self.parser.par_multi, freq_var, calculated_params)

    def _run_nested(self, range_pars, freq_var, calculated_params):
        # No more parameters to loop over.
        if not range_pars:
            self._run_kernel(freq_var, calculated_params)
        else:
            par = range_pars[0]

            # Loop over all values of a specific parameter.
            for val in self.options.__dict__[par]:
                self._sim_sym[par] = self.float(val)
                if par in self.global_vars:
                    cuda.memcpy_htod(self._gpu_sym[par], self.float(val))
                self._run_nested(range_pars[1:], freq_var, calculated_params)

    def _run_kernel(self, freq_var, calculated_params):
        # Calculate period and step size.
        if freq_var is not None:
            period = 2.0 * math.pi / self._sim_sym[freq_var]
        else:
            period = 1.0
        self.dt = self.float(period / self.options.spp)
        cuda.memcpy_htod(self._gpu_sym['dt'], self.dt)

        calculated_params(self)

        # Reinitialize the positions.
        self.vec = []
        for i in range(0, self.num_vars):
            # TODO: The arguments should include the current values of the
            # system parameters.
            vt = self.init_vectors(self, i).astype(self.float)
            self.vec.append(vt)
            cuda.memcpy_htod(self._gpu_vec[i], vt)

        kernel_args = [self._gpu_rng_state] + self._gpu_vec
        if self.scan_var is not None:
            kernel_args += [self._gpu_sv]

        # Prepare an array for number of periods for periodic variables.
        self.vec_nx = {}
        for i, v in self.periodic_map.iteritems():
            self.vec_nx[i] = numpy.zeros_like(self.vec[i]).astype(numpy.int64)

        # Prepare an array for initial value of the variables (after
        # transients).
        if self.options.omode == 'path':
            transient = False
            every = True
        else:
            transient = True
            every = False
            self.vec_start = []
            for i in range(0, self.num_vars):
                self.vec_start.append(numpy.zeros_like(self.vec[i]))

        def fold_variables(iter_, need_copy):
            for i, (period, freq) in self.periodic_map.iteritems():
                if iter_ % freq == 0:
                    if need_copy:
                        cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])

                    self.vec_nx[i] = numpy.add(self.vec_nx[i],
                            numpy.floor_divide(self.vec[i], period).astype(numpy.int64))
                    self.vec[i] = numpy.remainder(self.vec[i], period)
                    cuda.memcpy_htod(self._gpu_vec[i], self.vec[i])

        # Actually run the simulation here.
        for j in xrange(0, int(self.options.simperiods * self.options.spp / self.options.samples)+1):
            self.sim_t = self.options.samples * j * self.dt
            args = kernel_args + [numpy.float32(self.sim_t)]
            self.advance_sim.prepared_call((self.num_threads/self.block_size, 1), *args)

            if every:
                fold_variables(j, True)
                self.output_current()
            elif transient and self.sim_t >= self.options.transients * period:
                for i in range(0, self.num_vars):
                    cuda.memcpy_dtoh(self.vec_start[i], self._gpu_vec[i])
                transient = False
                self.start_t = self.sim_t
                self.vec_start_nx = self.vec_nx.copy()

            fold_variables(j, True)
            self.sim_t += self.options.samples * self.dt

        if not every:
            self.output_summary()

        self.output.finish_block()

    def output_current(self):
        vars = {}

        for i in self.req_vars:
            cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])
            vars[i] = self.get_var(i)

        self._output_results(vars, self.sim_t)

    def output_summary(self):
        vars = {}

        for i in self.req_vars:
            cuda.memcpy_dtoh(self.vec[i], self._gpu_vec[i])
            vars[i] = (self.get_var(i, True), self.get_var(i))

        self._output_results(vars)

    def _output_results(self, vars, *misc_pars):
        for i in range(0, self.scan_var_size):
            out = []

            for par in self.parser.par_multi:
                out.append(self._sim_sym[par])
            out.extend(misc_pars)

            if self.scan_var is not None:
                out.append(self._sv[i*self.options.paths])

            for func, req_vars in self.req_output:
                args = map(lambda x: vars[x], req_vars)
                if args and type(args[0]) is tuple:
                    args = map(lambda x:
                        (x[0][i*self.options.paths:(i+1)*self.options.paths],
                         x[1][i*self.options.paths:(i+1)*self.options.paths]), args)
                else:
                    args = map(lambda x:
                            x[i*self.options.paths:(i+1)*self.options.paths],
                            args)

                out.extend(func(self, *args))

            self.output.data(out)

