#!/usr/bin/python -u

import math
import numpy
import scipy
import sde
import sys

def init_vector(sdei, i):
    if i == 0:
        # Positions.
        a = numpy.zeros(sdei.num_threads)
        force = sdei.get_param('force').copy()
        force[(force > 1.0)] = 1.0
        a[:] = numpy.arcsin(force / sdei.get_param('psd'))
        return a
    else:
        # Velocities.
        a = numpy.zeros(sdei.num_threads)
        return a

def calculated_params(sdei):
    gam = sdei.get_param('gamma_')
    sdei.set_param('ns', numpy.float32(math.sqrt(sdei.options.d0 * sdei.dt * 2.0 * gam)))

def max_min(sdei, x):
    return [numpy.min(x), numpy.max(x), numpy.average(x)]

def diffusion_coefficient(sdei, x):
    ret = []
    x = x.astype(numpy.float64)
    deff1 = numpy.average(numpy.square(x)) - numpy.average(x)**2
    return [deff1 / (2.0 * sdei.sim_t)]

def myhist(sdei, x):
    dev = numpy.std(x)
    avg = numpy.average(x)

    bin_size = 0.02

    min_ = avg-3*dev
    max_ = avg+3*dev

    bins = numpy.arange(math.floor(min_ / bin_size) * bin_size, max_, bin_size)

    hst, edges = numpy.histogram(x, bins=bins, normed=False)
    ret = [[bins[0], bins[-1]]]
    for i in hst:
        ret.append([i])

    return ret


params = [('force', 'DC bias current'),
          ('gamma_', 'damping constant'),
          ('psd', 'potential strenght'),
          ('d0', 'noise strength')]

global_vars = ['ns']

code = """
    dx0 = x1;
    dx1 = -psd * sinf(x0) - gamma_ * x1 + force;
"""

noise_map = {1: ['ns']}
period_map = {0: (2.0 * numpy.pi, 1)}

sde_ = sde.SDE(code, params, global_vars, 2, 1, noise_map, period_map)
if not sde_.parse_args():
    sys.exit(1)

output = {
        'summary': {
            'main': [(sde.diffusion_coefficient, [0])],
        },
        'path': {
            'main': [(diffusion_coefficient, [0]), (max_min, [0])],
            'vhist': [(myhist, [1])],
        }
    }

sde_.prepare(sde.SRK2, init_vector)
sde_.simulate(output, calculated_params)

