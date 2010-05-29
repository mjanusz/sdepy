#!/usr/bin/python
#
# Three identical Josephson junctions coupled via a quasiparticle
# current/resistive shunt.
#
import math
import numpy
import sde
import sympy
import sys

def init_vector(sdei, i):
    return numpy.random.uniform(0.0, 2.0 * numpy.pi, sdei.num_threads)

sim_params = {'i1': 'constant force on the 1st particle',
              'i2': 'constant force on the 2nd particle',
              'i3': 'constant force on the 3rd particle',
              'alpha': 'coupling strength',
              'a1': 'AC drive amplitude for the 1st particle',
              'a2': 'AC drive amplitude for the 2nd particle',
              'a3': 'AC drive amplitude for the 3rd particle',
              'omega': 'AC drive frequency'}

# TODO: Fix this.
local_vars = {
        'ns0': lambda sdei: sympy.sqrt(sdei.S.d0 * sdei.S.dt),
        }

code = """
    dx0 = i1 - sinf(x0) + alpha * (i2 - sinf(x1)) + (a1 + alpha * a2) * cosf(omega * t);
    dx1 = i2 - sinf(x1) + alpha * (i1 - sinf(x0)) + alpha * (i3 - sinf(x2)) + (alpha * (a1 + a3) + a2) * cosf(omega * t);
    dx2 = i3 - sinf(x2) + alpha * (i2 - sinf(x1)) + (a3 + alpha * a2) * cosf(omega * t);
"""

ns_map = {0: ['ns0'], 1: ['ns0']}
period_map = {0: sde.PeriodInfo(period=2.0 * math.pi, freq=1),
              1: sde.PeriodInfo(period=2.0 * math.pi, freq=1),
              2: sde.PeriodInfo(period=2.0 * math.pi, freq=1)}

sdei = sde.SDE(code, sim_params, num_vars=3, num_noises=1, noise_map=ns_map, period_map=period_map,
               local_vars=local_vars)

output = {'path': {
            'main': [sde.OutputDecl(func=sde.avg_moments, vars=[0, 1, 2])],
            },
          'summary': {
            'main': [sde.OutputDecl(func=sde.drift_velocity, vars=[0, 1, 2])],
            }
        }

sdei.prepare(sde.SRK2, init_vector, freq_var='omega')
sdei.simulate(output)

