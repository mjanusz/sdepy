<%include file="rng.mako"/>
<%
	import math
%>

// Constants
%for param in const_parameters:
	__constant__ float ${param} = 0.0f;
%endfor

__constant__ float dt = 0.0f;
__constant__ unsigned int samples = 0;

<%def name="rng_uni()">
	rng_${rng}(
	%for i in range(0, rng_state_size):
		%if i > 0:
			,
		%endif
		lrng_state + ${i}
	%endfor
	)
</%def>

// System of differential equations to solve.
__device__ inline void RHS(
	%for i in range(0, rhs_vars):
		float &dx${i}, float x${i},
	%endfor
	%for param in par_cuda:
		float ${param},
	%endfor
		float t
)
{
	${sde_code}
}

<%def name="Euler()">
	for (i = 1; i <= samples; i++) {
		## RNG call.
		%for i in range(0, int(math.ceil(num_noises/2.0))):
			## If need an odd number of normal variates, then
			## every other iteration we can simply reuse one
			## variate from the previous one.
			%if num_noises % 2 and i == num_noises / 2:
				if (!(i & 1)) {
					n${2*i} = n${2*i+1};
				} else {
					n${2*i} = ${rng_uni()};
					n${2*i+1} = ${rng_uni()};
					bm_trans(n${2*i}, n${2*i+1});
				}
			%else:
				n${2*i} = ${rng_uni()};
				n${2*i+1} = ${rng_uni()};
				bm_trans(n${2*i}, n${2*i+1});
			%endif
		%endfor

		RHS(
			%for i in range(0, rhs_vars):
				xt${i}, x${i},
			%endfor
			%for param in par_cuda:
				${param},
			%endfor
			t
		);

		## Propagation.
		%for i in range(0, rhs_vars):
			x${i} = x${i} + xt${i} * dt
			%if i in noise_strength_map:
				%for j, n in enumerate(noise_strength_map[i]):
					%if n != 0.0:
						+ ${n}*n${j};
					%endif
				%endfor
			%endif
			;
		%endfor
		t = ct + i*dt;
	}
</%def>

<%def name="Milstein()">
	for (i = 1; i <= samples; i++) {
		## RNG call.
		%for i in range(0, int(math.ceil(num_noises/2.0))):
			## If need an odd number of normal variates, then
			## every other iteration we can simply reuse one
			## variate from the previous one.
			%if num_noises % 2 and i == num_noises / 2:
				if (!(i & 1)) {
					n${2*i} = n${2*i+1};
				} else {
					n${2*i} = ${rng_uni()};
					n${2*i+1} = ${rng_uni()};
					bm_trans(n${2*i}, n${2*i+1});
				}
			%else:
				n${2*i} = ${rng_uni()};
				n${2*i+1} = ${rng_uni()};
				bm_trans(n${2*i}, n${2*i+1});
			%endif
		%endfor

		RHS(
			%for i in range(0, rhs_vars):
				xt${i}, x${i},
			%endfor
			%for param in par_cuda:
				${param},
			%endfor
			t
		);

		%for i in range(0, rhs_vars):
			%if i in noise_strength_map:
				%for j, noise in enumerate(noise_strength_map[i]):
					float ${noise} = ${local_vars[noise]};
					%if ('d%s' % noise) not in local_vars:
						%if ('d%s' % noise) not in const_parameters:
							<%
								raise ValueError('Noise strength derivative d%s needs to be defined.' % noise)
							%>
						%endif
					%else:
						float d${noise} = ${local_vars['d%s' % noise]};
					%endif
				%endfor
			%endif
		%endfor

		## Propagation.
		%for i in range(0, rhs_vars):
			x${i} = x${i} + xt${i} * dt
			%if i in noise_strength_map:
				%for j, n in enumerate(noise_strength_map[i]):
					%if n != 0.0:
						+ ${n}*n${j} + 0.5f * ${n} * d${n} * (n${j}*n${j} - dt)
					%endif
				%endfor
			%endif
			;
		%endfor
		t = ct + i*dt;
	}
</%def>

<%def name="SRK2()">
	%for i in range(0, rhs_vars):
		float xtt${i}, xim${i};
	%endfor

	for (i = 1; i <= samples; i++) {
		## RNG call.
		%for i in range(0, int(math.ceil(num_noises/2.0))):
			## If we need an odd number of normal variates, then
			## every other iteration we can simply reuse one
			## variate from the previous one.
			%if num_noises % 2 and i == num_noises / 2:
				if (!(i & 1)) {
					n${2*i} = n${2*i+1};
				} else {
					n${2*i} = ${rng_uni()};
					n${2*i+1} = ${rng_uni()};
					bm_trans(n${2*i}, n${2*i+1});
				}
			%else:
				n${2*i} = ${rng_uni()};
				n${2*i+1} = ${rng_uni()};
				bm_trans(n${2*i}, n${2*i+1});
			%endif
		%endfor

		## First call to RHS.
		RHS(
			%for i in range(0, rhs_vars):
				xt${i}, x${i},
			%endfor
			%for param in par_cuda:
				${param},
			%endfor
			t
		);

		## Propagation.
		%for i in range(0, rhs_vars):
			xim${i} = x${i} + xt${i} * dt
			%if i in noise_strength_map:
				%for j, n in enumerate(noise_strength_map[i]):
					%if n != 0.0:
						+ ${n}*n${j};
					%endif
				%endfor
			%endif
			;
		%endfor
		t = ct + i*dt;

		## Second call to RHS.
		RHS(
			%for i in range(0, rhs_vars):
				xtt${i}, xim${i},
			%endfor
			%for param in par_cuda:
				${param},
			%endfor
			t
		);

		## Propagation.
		%for i in range(0, rhs_vars):
			x${i} += 0.5f*dt * (xt${i} + xtt${i})
			%if i in noise_strength_map:
				%for j, n in enumerate(noise_strength_map[i]):
					%if n != 0.0:
						+ ${n}*n${j}
					%endif
				%endfor
			%endif
			;
		%endfor
	}
</%def>

<%def name="rng_test()">
	x0 = ${rng_uni()};
</%def>

__global__ void AdvanceSim(unsigned int *rng_state,
	## Kernel arguments.
	%for i in range(0, rhs_vars):
		float *cx${i},
	%endfor
	%for param in par_cuda:
		float *c${param},
	%endfor
	float ct)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
## Local variables
	float
	%for i in range(0, rhs_vars):
		x${i}, xt${i},
	%endfor
	%for param in par_cuda:
		${param},
	%endfor
	%for i in range(0, num_noises + num_noises % 2):
		n${i},
	%endfor
	t;

	unsigned int lrng_state[${rng_state_size}];

	## Cache to local variables.
	%for i in range(0, rhs_vars):
		x${i} = cx${i}[idx];
	%endfor
	%for param in par_cuda:
		${param} = c${param}[idx];
	%endfor

	t = ct;

	%for i in range(0, rng_state_size):
		lrng_state[${i}] = rng_state[${rng_state_size} * idx + ${i}];
	%endfor

	// Additional local variables that depend on changing parameters.
	%for name, value in local_vars.iteritems():
		%if type(value) is not tuple:
			float ${name} = ${value};
		%endif
	%endfor

	%if method == 'SRK2':
		${SRK2()}
	%elif method == 'Euler':
		${Euler()}
	%elif method == 'Milstein':
		${Milstein()}
	%endif
##	${rng_test()}

	%for i in range(0, rhs_vars):
		cx${i}[idx] = x${i};
	%endfor

	%for i in range (0, rng_state_size):
		rng_state[${rng_state_size} * idx + ${i}] = lrng_state[${i}];
	%endfor
}

