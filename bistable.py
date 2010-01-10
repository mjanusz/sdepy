#!/usr/bin/python -u

import math
import numpy
import sde
import sys

def init_vector(sdei, i):
    if i == 0:
        # Positions.
        return numpy.zeros(sdei.num_threads)
#        return numpy.random.uniform(0, 2.0*math.pi, sdei.num_threads)
    else:
        # Velocities.
        a = numpy.zeros(sdei.num_threads)
        a[:] = 3.0
#        return numpy.zeros(sdei.num_threads)
        return a

def calculated_params(sdei):
    gam = sdei.get_param('gamma_')
    sdei.set_param('ns', numpy.float32(math.sqrt(sdei.options.d0 * sdei.dt * 2.0 * gam)))

sim_params = [('force', 'DC bias current'),
              ('gamma_', 'damping constant'),
              ('psd', 'potential strenght'),
              ('d0', 'noise strength')]

mod_vars = ['ns']

code = """
    dx0 = x1;
    dx1 = -psd * sinf(x0) - gamma_ * x1 + force;
"""

ns_map = [[0], ['ns']];

sdei = sde.SDE(sim_params, mod_vars)
if not sdei.parse_args():
    sys.exit(1)

sdei.cuda_prep_gen(2, init_vector, 1, ns_map, code)
sdei.cuda_run(64, calculated_params)



