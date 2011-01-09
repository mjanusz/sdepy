from collections import Iterable
import operator
import sys
import numpy

def drift_velocity(sde, *args):
    ret = []
    for starting, final in args:
        a = starting.astype(numpy.float64)
        b = final.astype(numpy.float64)

        ret.append((numpy.average(b) - numpy.average(a)) /
                (sde.sim_t - sde.start_t))

    return ret

def abs_drift_velocity(sde, *args):
    ret = []
    for starting, final in args:
        a = starting.astype(numpy.float64)
        b = final.astype(numpy.float64)

        ret.append(numpy.average(numpy.abs(b - a)) /
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

class TextOutput(object):
    def __init__(self, sde, subfiles):
        self.sde = sde
        self.subfiles = subfiles
        self.out = {}

        if sde.options.output is not None:
            self.out['main'] = open(sde.options.output, 'w')

            for sub in subfiles:
                if sub == 'main':
                    continue
                self.out[sub] = open('%s_%s' % (sde.options.output, sub), 'w')
        else:
            if len(subfiles) > 1:
                raise ValueError('Output file name required so that auxiliary data stream can be saved.')

            self.out['main'] = sys.stdout

    def finish_block(self):
        print >>self.out['main'], ''

    def flush(self):
        self.out['main'].flush()

    def data(self, **kwargs):
        for name, val in kwargs.iteritems():
            def my_rep(val):
                if isinstance(val, Iterable):
                    return ' '.join(my_rep(x) for x in val)
                else:
                    return str(val)

            rep = [my_rep(x) for x in val]
            print >>self.out[name], ' '.join(rep)

    def header(self):
        out = self.out['main']

        print >>out, '# %s' % ' '.join(sys.argv)
        if self.sde.options.seed is not None:
            print >>out, '# seed = %d' % self.sde.options.seed
        print >>out, '# sim periods = %d' % self.sde.options.simperiods
        print >>out, '# transient periods = %d' % self.sde.options.transients
        print >>out, '# samples = %d' % self.sde.options.samples
        print >>out, '# paths = %d' % self.sde.options.paths
        print >>out, '# spp = %d' % self.sde.options.spp
        for par in self.sde.parser.par_single:
            print >>out, '# %s = %f' % (par, self.sde.options.__dict__[par])
        for par in self.sde.par_multi_ordered:
            print >>out, '# %s = %s' % (par, ' '.join(str(x) for x in self.sde.options.__dict__[par]))
        for par in self.sde.scan_vars:
            print >>out, '# %s = %s' % (par, ' '.join(str(x) for x in self.sde.options.__dict__[par]))

    def close(self):
        pass

class NpyOutput(object):
    def __init__(self, sde, subfiles):
        self.sde = sde
        self.subfiles = subfiles

        # This dictionary maps the subfile name to a list of lists.
        # Every entry in the outer list represents one point in the
        # parameter space.  The inner list represents the results for
        # a particular set of parameters.
        self.cache = {}

    def finish_block(self):
        pass

    def flush(self):
        self.close()

    def data(self, **kwargs):
        for name, val in kwargs.iteritems():
            self.cache.setdefault(name, []).append(val)

    def header(self):
        self.cmdline = '%s' % ' '.join(sys.argv)
        self.scan_vars = self.sde.scan_vars
        self.par_multi_ordered = self.sde.par_multi_ordered

    def close(self):
        out = {}
        shape = []

        for par in self.par_multi_ordered:
            out[par] = getattr(self.sde.options, par)
            shape.append(len(out[par]))

        for sv in self.scan_vars:
            out[sv] = getattr(self.sde.options, sv)
            shape.append(len(out[sv]))

        for name, val in self.cache.iteritems():
            inner_len = max(len(x) for x in val)
            out[name] = numpy.array(val, dtype=self.sde.float)

            if shape and reduce(operator.mul, shape) * inner_len == reduce(operator.mul, out[name].shape):
                out[name] = numpy.reshape(out[name], shape + [inner_len])

        numpy.savez(self.sde.options.output, cmdline=self.cmdline, scan_vars=self.scan_vars,
                    par_multi=self.par_multi_ordered, options=self.sde.options, **out)

class StoreOutput(object):
    def __init__(self, sde, subfiles):
        self.sde = sde
        self.subfiles = subfiles
        self.cache = {}

    def finish_block(self):
        pass

    def flush(self):
        pass

    def data(self, **kwargs):
        for name, val in kwargs.iteritems():
            self.cache.setdefault(name, []).append(val)

    def header(self):
        self.scan_vars = self.sde.scan_vars
        self.par_multi_ordered = self.sde.par_multi_ordered

    def close(self):
        out = {}
        shape = []

        for par in self.par_multi_ordered:
            out[par] = getattr(self.sde.options, par)
            shape.append(len(out[par]))

        for sv in self.scan_vars:
            out[sv] = getattr(self.sde.options, sv)
            shape.append(len(out[sv]))

        for name, val in self.cache.iteritems():
            inner_len = max(len(x) for x in val)
            out[name] = numpy.array(val, dtype=self.sde.float)

            if shape and reduce(operator.mul, shape) * inner_len == reduce(operator.mul, out[name].shape):
                out[name] = numpy.reshape(out[name], shape + [inner_len])

        self.out = out
