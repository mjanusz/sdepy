#!/usr/bin/python

import scipy as sp
import numpy as np
import matplotlib
matplotlib.use('GTKAgg')

from scipy.integrate import odeint, ode
from pylab import *

import sympy
import sde

def init_vector(sdei, i):
    if i == 0:
        return np.random.uniform(0.0, 2.0 * math.pi, sdei.num_threads)
    else:
        return np.random.uniform(-2.0, 2.0, sdei.num_threads)

sim_params = {'force': 'DC bias current',
              'gam': 'damping constant',
              'omega': 'AC drive frequency',
              'd0': 'noise strength',
              'amp': 'AC drive amplitude'}

local_vars = { 'ns': lambda sdei: sympy.sqrt(sdei.S.d0 * sdei.S.dt * 2.0 * sdei.S.gam) }

code = """
    dx0 = x1;
    dx1 = -sinf(x0) + amp * cosf(omega * t) + force - gam * x1;
"""

ns_map = {1: ['ns']}
period_map = {0: sde.PeriodInfo(period=2.0 * math.pi, freq=1)}

sdei = sde.SDE(code, sim_params, num_vars=2, num_noises=1, noise_map=ns_map, period_map=period_map,
               local_vars=local_vars)

output = {'path': {
            'main': [sde.OutputDecl(func=sde.avg_moments, vars=[0,1])],
            },
          'summary': {
            'main': [sde.OutputDecl(func=sde.drift_velocity, vars=[0])],
#            'abs': [sde.OutputDecl(func=sde.abs_drift_velocity, vars=[0])],
            }
        }

sdei.prepare(sde.SRK2, init_vector, freq_var='omega')
sdei.simulate(output)

a = sdei.output.out['main']
plot(np.mod(a[:,1], 2 *pi), a[:,3], '.', ms=0.5)
#plot(a[:,0], a[:,1])
show()
