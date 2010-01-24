#ifndef __CU_RNG_H
#define __CU_RNG_H

#include <float.h>

#define PI 3.14159265358979f

/*
 * Return a uniformly distributed random number from the
 * [0;1] range.
 */

%if rng == 'xorshift32':
__device__ float rng_xorshift32(unsigned int *state)
{
	unsigned int x = *state;

	x = x ^ (x >> 13);
	x = x ^ (x << 17);
	x = x ^ (x >> 5);

	*state = x;

	return x * 2.328306e-10f;
//	return x / 4294967296.0f;
}
%endif

%if rng == 'kiss32':
__device__ float rng_kiss32(unsigned int *x, unsigned int *y,
		unsigned int *z, unsigned int *w)
{
	*x = 69069 * *x + 1234567;		// CONG
	*z = 36969 * (*z & 65535) + (*z >> 16);	// znew
	*w = 18000 * (*w & 65535) + (*w >> 16);	// wnew  & 6553?
	*y ^= (*y << 17);			// SHR3
	*y ^= (*y >> 13);
	*y ^= (*y << 5);

	return ((((*z << 16) + *w) ^ *x) + *y) * 2.328306e-10f;	// (MWC ^ CONG) + SHR3
}
%endif

%if rng == 'nr32':
// 32-bit PRNG from Numerical Recipes, 3rd Ed.
// Period: 3.11 * 10^37
__device__ float rng_nr32(unsigned int *u, unsigned int *v,
		unsigned int *w1, unsigned int *w2)
{
	unsigned int x, y;

	*u = *u * 2891336453U + 1640531513U;
	*v ^= *v >> 13;
	*v ^= *v << 17;
	*v ^= *v >> 5;
	*w1 = 33378 * (*w1 & 0xffff) + (*w1 >> 16);
	*w2 = 57225 * (*w2 & 0xffff) + (*w2 >> 16);

	x = *u ^ (*u << 9);
	x ^= x >> 17;
	x ^= x << 6;

	y = *w1 ^ (*w1 << 17);
	y ^= y >> 15;
	y ^= y << 5;
	return ((x + *v) ^ (y + *w2)) * 2.328306e-10f;
}
%endif

/*
 * Generate two normal variates given two uniform variates.
 */
__device__ void bm_trans(float& u1, float& u2)
{
	float r = sqrtf(-2.0f * logf(u1 + FLT_EPSILON));
	float phi = 2.0f * PI * u2;
	u1 = r * cosf(phi);
	u2 = r * sinf(phi);
}

#endif /* __CU_RNG_H */
