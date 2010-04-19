#!/usr/bin/python -u

import math
import numpy
import scipy
import sde
import sympy
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

def max_min(sdei, x):
    return [numpy.min(x), numpy.max(x), numpy.average(x)]

def diffusion_coefficient(sdei, x):
    ret = []
    x = x.astype(numpy.float64)
    deff1 = numpy.average(numpy.square(x)) - numpy.average(x)**2
    return [deff1 / (2.0 * sdei.sim_t)]

def myhist(sdei, x):
    return x

params = {'force': 'DC bias current',
          'gamma_': 'damping constant',
          'psd': 'potential strenght',
          'd0': 'noise strength'}

local_vars = { 'ns': lambda sdei: sympy.sqrt(sdei.S.d0 * sdei.S.dt * 2.0 * sdei.S.gamma_) }

code = """
    dx0 = x1;
    dx1 = -psd * sinf(x0) - gamma_ * x1 + force;
"""

noise_map = {1: ['ns']}
period_map = {0: sde.PeriodInfo(period=2.0 * numpy.pi, freq=1)}

sde_ = sde.SDE(code, params, num_vars=2, num_noises=1, noise_map=noise_map, period_map=period_map,
        local_vars=local_vars)

output = {
        'summary': {
            'main': [sde.OutputDecl(func=sde.diffusion_coefficient, vars=[0])],
        },
        'path': {
            'main': [sde.OutputDecl(func=diffusion_coefficient, vars=[0]),
                     sde.OutputDecl(func=max_min, vars=[0])],
            'vhist': [sde.OutputDecl(func=myhist, vars=[1])],
        }
    }

sde_.prepare(sde.SRK2, init_vector)
sde_.simulate(output)

