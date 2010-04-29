#!/usr/bin/python
#
# Two Brownian particles interacting via a Kuromoto-like potential.
#
import math
import numpy
import sde
import sympy
import sys

def init_vector(sdei, i):
    if i == 0:
        return numpy.random.uniform(0.0, 2.0 * numpy.pi, sdei.num_threads)
    else:
        return numpy.random.uniform(0.0, 2.0 * numpy.pi, sdei.num_threads)

sim_params = {'f1': 'constant force on the 1st particle',
              'f2': 'constant force on the 2nd particle',
              'gam1': 'damping for the 1st particle',
              'is': 'interaction strength',
              'd0': 'noise strength',
              'amp': 'AC drive amplitude'}

local_vars = { 
        'ns0': lambda sdei: sympy.sqrt(sdei.S.d0 * sdei.S.dt * 2.0 / sdei.S.gam1),
        'ns1': lambda sdei: sympy.sqrt(sdei.S.d0 * sdei.S.dt * 2.0 / sdei.S.gam1),
        }

code = """
    dx0 = (is * sinf(x0 - x1) + f1 + amp * cosf(t)) / gam1;
    dx1 = (-is * sinf(x0 - x1) + f2) / gam1;
"""

ns_map = {0: ['ns0', 0], 1: [0, 'ns1']}
period_map = {0: sde.PeriodInfo(period=2.0 * math.pi, freq=1.0)}

sdei = sde.SDE(code, sim_params, num_vars=2, num_noises=2, noise_map=ns_map, period_map=period_map,
               local_vars=local_vars)

output = {'path': {
            'main': [sde.OutputDecl(func=sde.avg_moments, vars=[0, 1])],
            },
          'summary': {
            'main': [sde.OutputDecl(func=sde.drift_velocity, vars=[0, 1])],
            }
        }

sdei.prepare(sde.SRK2, init_vector)
sdei.simulate(output)

