#!/usr/bin/python

import math
import numpy
import sde
import sys

def init_vector(sdei, i):
    if i == 0:
        # Positions.
        #       return numpy.random.uniform(0, 2.0*math.pi, sdei.num_threads)
        a = numpy.zeros(sdei.num_threads)
        a[:] = 0.5
        return a
    else:
        # Velocities.
        return numpy.zeros(sdei.num_threads)

def calculated_params(sdei):
    gam = sdei.get_param('gam')
    sdei.set_param('ns', numpy.float32(math.sqrt(sdei.options.d0 * sdei.dt * 2.0 * gam)))

sim_params = [('force', 'DC bias current'),
              ('gam', 'damping constant'),
              ('omega', 'AC drive frequency'),
              ('d0', 'noise strength'),
              ('amp', 'AC drive amplitude')]

global_vars = ['ns']

code = """
    dx0 = x1;
    dx1 = -2.0f * PI * cosf(2.0f * PI * x0) + amp * cosf(omega * t) + force - gam * x1;
"""

ns_map = {1: ['ns']}
period_map = {0: (1, 1)}

sdei = sde.SDE(code, sim_params, global_vars, 2, 1, ns_map, period_map)
if not sdei.parse_args():
    sys.exit(1)

output = {'path': {
            'main': [(sde.avg_moments, [0])],
            },
          'summary': {
            'main': [(sde.drift_velocity, [0])],
            }
        }

sdei.prepare(sde.SRK2, init_vector)
sdei.simulate(output, calculated_params, freq_var='omega')



