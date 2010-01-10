import math
import os
import pwd
import sys

from optparse import OptionParser, OptionValueError

import numpy

import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler

from mako.template import Template
from mako.lookup import TemplateLookup


def _parse_range(option, opt_str, value, parser):
    vals = value.split(',')

    if len(vals) == 1:
        setattr(parser.values, option.dest, numpy.float32(value))
        parser.par_single.append(option.dest)
    elif len(vals) == 3:
        start = float(vals[0])
        stop = float(vals[1])
        step = (stop-start) / (int(vals[2])-1)
        setattr(parser.values, option.dest, numpy.arange(start, stop+0.999*step, step, numpy.float32))
        parser.par_multi.append(option.dest)
    else:
        raise OptionValueError('"%s" has to be a single value or a range of the form \'start,stop,steps\'' % opt_str)

def _get_module_source(*files):
    """Load multiple .cu files and join them into a string."""
    src = ""
    for fname in files:
        fp = open(fname)
        src += fp.read()
        fp.close()
    return src


class SDESolverGenerator(object):
    def __init__(self, noises, noise_strength_map, num_vars, parameters, sde_code):
        self.noises = noises
        self.vars = num_vars
        self.parameters = parameters
        self.sde_code = sde_code
        self.noise_strength_map = noise_strength_map

        self.noise_strengths = set()
        for i in noise_strength_map:
            self.noise_strengths.update(set(i))
        self.noise_strengths.difference_update(set([0]))

    def SRK2(self, par_cuda):
        """Return SRK2 code for the SDE.

        par_cuda: set of parameter names which are to have multiple values
            in a single CUDA kernel call.
        """
        num_noises = self.noises + self.noises % 2

        ctx = {}
        ctx['const_parameters'] = self.parameters - par_cuda | set(self.noise_strengths)
        ctx['par_cuda'] = par_cuda
        ctx['rhs_vars'] = self.vars
        ctx['noise_strength_map'] = self.noise_strength_map
        ctx['noises'] = self.noises
        ctx['num_noises'] = num_noises
        ctx['sde_code'] = self.sde_code

        lookup = TemplateLookup(directories=sys.path, module_directory='/tmp/sailfish_modules-%s' %
                                (pwd.getpwuid(os.getuid())[0]))
        sde_template = lookup.get_template('sde.mako')
        return sde_template.render(**ctx)

class SDE(object):
    def __init__(self, sim_params, global_vars):
        """sim_params: list of simulation parameters defined as tuples (param name, param description)
        global_vars: list of global symbols in the CUDA code
        """
        self.parser = OptionParser()
        self.parser.add_option('--spp', dest='spp', help='steps per period', metavar='DT', type='int', action='store', default=100)
        self.parser.add_option('--samples', dest='samples', help='sample the position every N steps', metavar='N', type='int', action='store', default=100)
        self.parser.add_option('--paths', dest='paths', help='number of paths to sample', type='int', action='store', default=256)
        self.parser.add_option('--transients', dest='transients', help='number of periods to ignore because of transients', type='int', action='store', default=200)
        self.parser.add_option('--simperiods', dest='simperiods', help='number of periods in the simulation', type='int', action='store', default=2000)
        self.parser.add_option('--output_mode', dest='omode', help='output mode', type='choice', choices=['avgv', 'avgpath'], action='store', default='avgv')
        self.parser.add_option('--seed', dest='seed', help='RNG seed', type='int', action='store', default=None)
        self.parser.add_option('--output_format', dest='oformat', help='output file format', type='choice', choices=['text', 'hdf_expanded', 'hdf_nested'], action='store', default='text')
        self.parser.add_option('--save_src', dest='save_src', help='save the generated source to FILE', metavar='FILE',
                               type='string', action='store', default=None)

        self.parser.par_multi = []
        self.parser.par_single = []
        self.sim_params = sim_params
        self.global_vars = global_vars

        self._sim_sym = {}
        self._gpu_sym = {}

        global_vars.append('samples');
        global_vars.append('dt');
        global_vars.extend(sim_params)

        for name, help_string in sim_params:
            self.parser.add_option('--%s' % name, dest=name, action='callback',
                    callback=_parse_range, type='string', help=help_string,
                    default=None)

    def parse_args(self):
        self.options, self.args = self.parser.parse_args()
        opt_ok = True
        for name, hs in self.sim_params:
            if self.options.__dict__[name] == None:
                print 'Required option "%s" not specified.' % name
                opt_ok = False

        return opt_ok

    # TODO: add support for scanning over a set of parameters
    def _find_cuda_scan_par(self):
        """Automatically determine which parameter is to be scanned over
        in a single kernel run.
        """
        max_i = 0
        max_par = None
        max = 0

        # Look for a parameter with the highest number of values.
        for i, par in enumerate(self.parser.par_multi):
            if len(self.options.__dict__[par]) > max:
                max_par = par
                max = len(self.options.__dict__[par])
                max_i = i

        # Remove this parameter from the list of parameters with multiple
        # values, so that we don't loop over it when running the simulation.
        del self.parser.par_multi[max_i]
        self.global_vars.remove(max_par)

        return (max_par, (self.options.__dict__[par][0],
                          self.options.__dict__[par][-1],
                          max))

    def cuda_prep_gen(self, num_vars, init_vector, noises, noise_strength_map,
                      sde_code):
        """Prepare a SDE simulation given only the SDE code.  This method
        automatically generates the solver code for the provided SDE.

        num_vars: number of independent variables in the SDE
        init_vector: see cuda_prep() description
        noises: number of white noise terms
        noise_strength_map: list of 'num_vars' elements.  Each element is a list
            of 'noises' elements.  The (j,i)-th element is either zero, or the name of
            a constant CUDA variable by which the i-th noise term is to be multiplied
            when being added to the j-th derivative term.
        sde_code: code to calculate the deterministic part of the derivatives
        """

        # Make sure the noise strength map is correctly defined.
        assert len(noise_strength_map) == num_vars
        for i in noise_strength_map:
            assert len(i) == noises

        gen = SDESolverGenerator(noises, noise_strength_map, num_vars,
                                 set(self.global_vars) - set(['dt', 'samples']),
                                 sde_code)
        scan_var, scan_var_range = self._find_cuda_scan_par()
        src = gen.SRK2(set([scan_var]))

        if self.options.save_src is not None:
            with open(self.options.save_src, 'w') as file:
                print >>file, src

        return self.cuda_prep(num_vars, init_vector, src, scan_var, scan_var_range, sim_func='AdvanceSim')

    def cuda_prep(self, num_vars, init_vector, sources, scan_var,
                  scan_var_range, sim_func='advanceSystem'):
        """Prepare a SDE simulation for execution using CUDA.

        num_vars: number of variables in the SDE system
        init_vector: a function which takes an instance of this class and the
            variable number as arguments and returns an initialized vector of
            size num_threads
        sources: list of source code files for the simulation
        scan_var_range: a tuple (start, end, steps) specifying the values of
            the scan variable.  The scan variable is a system parameter for
            whose multiple values results are obtained in a single CUDA kernel
            launch)
        sim_func: name of the kernel advacing the simulation in time
        """

        if self.options.seed is not None:
            numpy.random.seed(self.options.seed)

        self.scan_var = scan_var
        self.sv_start, self.sv_end, self.sv_samples = scan_var_range
        self._sim_prep_mod(sources, init_vector, sim_func)
        self._sim_prep_const()
        self._sim_prep_var(num_vars)
        self._output_header()

    def _sim_prep_mod(self, sources, init_vector, sim_func):
        self.num_threads = self.sv_samples * self.options.paths
        self.init_vector = init_vector

        if type(sources) is str:
            self.mod = pycuda.compiler.SourceModule(sources, options=['--use_fast_math'])
        else:
            self.mod = pycuda.compiler.SourceModule(_get_module_source(*sources), options=['--use_fast_math'])
        self.advance_sim = self.mod.get_function(sim_func)

    def _sim_prep_const(self):
        # Const variables initialization
        for var in self.global_vars:
            self._gpu_sym[var] = self.mod.get_global(var)[0]

        # Simulation parameters
        samples = numpy.uint32(self.options.samples)
        cuda.memcpy_htod(self._gpu_sym['samples'], samples)

        # Parameters which only have a single value
        for par in self.parser.par_single:
            self._sim_sym[par] = self.options.__dict__[par]

            # If a variable is not in the dictionary, then it is automatically
            # calculated and will be set at a later stage.
            if par in self._gpu_sym:
                cuda.memcpy_htod(self._gpu_sym[par], self.options.__dict__[par])

    def _sim_prep_var(self, num_vars):
        self._vec = []
        self._gpu_vec = []
        self._num_vars = num_vars

        # Prepare device vectors.
        for i in range(0, self._num_vars):
            vt = self.init_vector(self, i).astype(numpy.float32)
            self._vec.append(vt)
            self._gpu_vec.append(cuda.mem_alloc(vt.nbytes))

        # Initialize the RNG seeds.
        self._rng_state = numpy.random.randint(0, sys.maxint, self.num_threads)
        self._rng_state = self._rng_state.astype(numpy.uint32)
        self._gpu_rng_state = cuda.mem_alloc(self._rng_state.nbytes)
        cuda.memcpy_htod(self._gpu_rng_state, self._rng_state)

        # Initialize the scan variable.
        tmp = []
        if self.sv_samples == 1:
            tmp = [self.sv_start, ] * self.options.paths
        else:
            eps = (self.sv_end - self.sv_start) / (self.sv_samples-1)
            for i in numpy.arange(self.sv_start, self.sv_end + eps, eps):
                tmp.extend([i, ] * self.options.paths)

        self._sv = numpy.array(tmp, dtype=numpy.float32)
        self._gpu_sv = cuda.mem_alloc(self._sv.nbytes)
        cuda.memcpy_htod(self._gpu_sv, self._sv)

    def cuda_run(self, block_size, freq_var, calculated_params):
        """Run a CUDA SDE simulation.

        block_size: CUDA block size
        freq_var: name of the parameter which is to be interpreted as a
            frequency (determines the step size 'dt')
        calculated_params: a function, which given an instance of this
            class will setup the values of automatically calculated
            parameters
        """
        self.block_size = block_size
        arg_types = ['P'] + ['P']*self._num_vars + ['P', numpy.float32]
        self.advance_sim.prepare(arg_types, block=(block_size, 1, 1))
        self._run_nested(self.parser.par_multi, freq_var, calculated_params)

    def set_param(self, name, val):
        self._sim_sym[name] = val
        cuda.memcpy_htod(self._gpu_sym[name], val)

    def get_param(self, name):
        return self._sim_sym[name]

    def _run_kernel(self, freq_var, calculated_params):
        # Calculate period and step size.
        period = 2.0 * math.pi / self._sim_sym[freq_var]
        self.dt = numpy.float32(period / self.options.spp)
        cuda.memcpy_htod(self._gpu_sym['dt'], self.dt)

        calculated_params(self)

        # Reinitialize the positions.
        self._vec = []
        for i in range(0, self._num_vars):
            vt = self.init_vector(self, i).astype(numpy.float32)
            self._vec.append(vt)
            cuda.memcpy_htod(self._gpu_vec[i], vt)

        self.text_prefix = []
        for par in self.parser.par_multi:
            self.text_prefix.append(self._sim_sym[par])

        if self.options.omode == 'avgv':
            self._run_avgv(period)
        elif self.options.omode == 'avgpath':
            self._run_avgpath(period)

        self._output_finish_block()

    def _run_avgpath(self, period):
        for j in range(0, int(self.options.simperiods * self.options.spp / self.options.samples)+1):
            self.sim_t = self.options.samples * j * self.dt
            args = [self._gpu_rng_state] + self._gpu_vec + [self._gpu_sv, numpy.float32(self.sim_t)]
            self.advance_sim.prepared_call((self.num_threads/self.block_size, 1), *args)

            for i in range(0, self._num_vars):
                cuda.memcpy_dtoh(self._vec[i], self._gpu_vec[i])

            self._output_results(self._get_var_avgpath, self.sim_t)
            self.sim_t += self.options.samples * self.dt

    def _run_avgv(self, period):
        transient = True
        self._vec_start = []
        for i in range(0, self._num_vars):
            self._vec_start.append(numpy.zeros_like(self._vec[i]))

        # Actually run the simulation
        for j in range(0, int(self.options.simperiods * self.options.spp / self.options.samples)+1):
            self.sim_t = self.options.samples * j * self.dt
            if transient and self.sim_t >= self.options.transients * period:
                for i in range(0, self._num_vars):
                    cuda.memcpy_dtoh(self._vec_start[i], self._gpu_vec[i])
                transient = False
                self.start_t = self.sim_t

            args = [self._gpu_rng_state] + self._gpu_vec + [self._gpu_sv, numpy.float32(self.sim_t)]
            self.advance_sim.prepared_call((self.num_threads/self.block_size, 1), *args)

            self.sim_t += self.options.samples * self.dt

        for i in range(0, self._num_vars):
            cuda.memcpy_dtoh(self._vec[i], self._gpu_vec[i])

        self._output_results(self._get_var_avgv)

    def _get_var_avgpath(self, i, start, end):
        return numpy.average(self._vec[i][start:end])

    def _get_var_avgv(self, i, start, end):
        return (numpy.average(self._vec[i][start:end]) -
                numpy.average(self._vec_start[i][start:end])) / (self.sim_t - self.start_t)

    def _output_results(self, get_var, *misc_pars):
        for i in range(0, self.sv_samples):
            out = []
            out.extend(self.text_prefix)
            out.extend(misc_pars)
            out.append(self._sv[i*self.options.paths])
            for j in range(0, len(self._vec)):
                out.append(get_var(j, i*self.options.paths, (i+1)*self.options.paths))

            if self.options.oformat == 'text':
                self._output_text(out)
            elif self.options.oformat == 'hdf_expanded':
                self._output_hdfexp(out)

    def _run_nested(self, range_pars, freq_var, calculated_params):
        # No more parameters to loop over.
        if not range_pars:
            self._run_kernel(freq_var, calculated_params)
        else:
            par = range_pars[0]

            # Loop over all values of a specific parameter.
            for val in self.options.__dict__[par]:
                self._sim_sym[par] = val
                if par in self.global_vars:
                    cuda.memcpy_htod(self._gpu_sym[par], val)
                self._run_nested(range_pars[1:], freq_var, calculated_params)

    def _output_finish_block(self):
        if self.options.oformat == 'text':
            print
        elif self.options.oformat == 'hdf_expanded':
            self.h5table.flush()

    def _output_text(self, pars):
        rep = ['%12.5e' % x for x in pars]
        print ' '.join(rep)

    def _output_hdfexp(self, pars):
        record = self.h5table.row
        for i, col in enumerate(self.h5table.cols._v_colnames):
            record[col] = pars[i]
        record.append()

    def _output_header(self):
        if self.options.oformat == 'text':
            self._output_text_header()
        elif self.options.oformat == 'hdf_expanded':
            self._output_hdfexp_header()

    def _output_hdfexp_header(self):
        import tables
        desc = {}
        pars = []
        pars.extend(self.parser.par_multi)
        pars.append(self.scan_var)
        if self.options.omode == 'avgpath':
            pars.append('t')
        for i in range(0, self._num_vars):
            pars.append('x%d' % i)

        for i, par in enumerate(pars):
            desc[par] = tables.Float32Col(pos=i)

        self.h5file = tables.openFile('output.h5', mode = 'w')
        self.h5group = self.h5file.createGroup('/', 'results', 'simulation results')
        self.h5table = self.h5file.createTable(self.h5group, 'results', desc, 'results')

    def _output_text_header(self):
        if self.options.seed is not None:
            print '# seed = %d' % self.options.seed
        print '# sim periods = %d' % self.options.simperiods
        print '# transient periods = %d' % self.options.transients
        for par in self.parser.par_single:
            print '# %s = %f' % (par, self.options.__dict__[par])
        print '#',
        for par in self.parser.par_multi:
            print par,

        print '%s' % self.scan_var,
        for i in range(0, self._num_vars):
            print 'x%d' % i,

        print


