#!/usr/bin/python -u

import math
import numpy
import sde
import sys

def init_vector(sdei, i):
    if i == 0:
#        # Positions.
        a = numpy.zeros(sdei.num_threads)
        a[:] = numpy.arcsin(sdei.get_param('force') / sdei.get_param('psd'))
        return a
    else:
        # Velocities.
        a = numpy.zeros(sdei.num_threads)
        a[:] = 3.0
#        return numpy.zeros(sdei.num_threads)
        return a

def calculated_params(sdei):
    gam = sdei.get_param('gamma_')
    sdei.set_param('ns', numpy.float32(math.sqrt(sdei.options.d0 * sdei.dt * 2.0 * gam)))

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
period_map = {0: (2.0 * numpy.pi, 10)}

sde_ = sde.SDE(code, params, global_vars, 2, 1, noise_map, period_map)
if not sde_.parse_args():
    sys.exit(1)

output = {'path': [(sde.avg_moments, [0])],
          'summary': [(sde.diffusion_coefficient, [0])]}

sde_.prepare(sde.SRK2, init_vector)
sde_.simulate(output, calculated_params)


