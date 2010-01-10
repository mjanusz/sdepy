#!/usr/bin/python

import math
import numpy
import sde
import sys

def init_vector(sdei, i):
	if i == 0:
        # Positions.
		return numpy.random.uniform(0, 2.0*math.pi, sdei.num_threads)
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

mod_vars = ['ns']

code = """
	dx0 = x1;
	dx1 = -2.0f * PI * cosf(2.0f * PI * x0) + amp * cosf(omega * t) + force - gam * x1;
"""

ns_map = [[0], ['ns']];

sdei = sde.SDE(sim_params, mod_vars)
if not sdei.parse_args():
	sys.exit(1)

sdei.cuda_prep_gen(2, init_vector, 1, ns_map, code)
sdei.cuda_run(64, 'omega', calculated_params)



