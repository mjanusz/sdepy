<%include file="rng.mako"/>

// Constants
%for param in const_parameters:
	__constant__ float ${param} = 0.0f;
%endfor

__constant__ float dt = 0.0f;
__constant__ int samples = 0;

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

<%def name="SRK2()">
	for (i = 1; i <= samples; i++) {
		## RNG call.
		%for i in range(0, num_noises/2):
			n${2*i} = ${rng_uni()};
			n${2*i+1} = ${rng_uni()};
			bm_trans(n${2*i}, n${2*i+1});
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

		## RNG call.
		%for i in range(0, noises/2):
			n${2*i} = ${rng_uni()};
			n${2*i+1} = ${rng_uni()};
			bm_trans(n${2*i}, n${2*i+1});
		%endfor

		## Propagation.
		%for i in range(0, rhs_vars):
			x${i} += 0.5f*dt * (xt${i} + xtt${i})
			%if i in noise_strength_map:
				%for j, n in enumerate(noise_strength_map[i]):
					%if j == noises-1 and noises % 2:
						## Reuse a noise already calculated during the
						## first set of calls to bm_trans.
						+ ${n}*n${j+1}
					%else:
						+ ${n}*n${j}
					%endif
				%endfor
			%endif
			;
		%endfor
		t = ct + i*dt;
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
		x${i}, xt${i}, xtt${i}, xim${i},
	%endfor
	%for param in par_cuda:
		${param},
	%endfor
	%for i in range(0, num_noises):
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

	%for i in range (0, rng_state_size):
		lrng_state[${i}] = rng_state[${rng_state_size} * idx + ${i}];
	%endfor

	${SRK2()}
##	${rng_test()}

	%for i in range(0, rhs_vars):
		cx${i}[idx] = x${i};
	%endfor

	%for i in range (0, rng_state_size):
		rng_state[${rng_state_size} * idx + ${i}] = lrng_state[${i}];
	%endfor
}

