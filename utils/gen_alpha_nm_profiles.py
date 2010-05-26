#!/usr/bin/python
#
# Given a list of (omega, a1) mobility scans, assemble a set of (omega, a1)
# pairs for which negative mobility of the 1st or 2nd particle has been
# detected, and then perform a scan over alpha for these points.
#
# ./gen_alpha_nm_profiles.py scan1.npz scan2.npz ...

from subprocess import Popen, PIPE
import os
import numpy as np
import sys

values = set()

for file in sys.argv[1:]:
    a = np.load(file)
    neg_mob_map = np.logical_or(a['main'][:,:,0] < -0.09, a['main'][:,:,1] < -0.09)
    yi, xi = np.nonzero(neg_mob_map)

    for i in range(0, len(xi)):
        x = xi[i]
        y = yi[i]
        values.add((a['omega'][y], a['a1'][x]))

print len(values), 'to scan'

for val in values:
    par_om, par_a1 = val

    fname = os.path.join('profiles',
                         'jj2_ident_anm_alpha_prof_i0.05_omega%s_a1%s' % (par_om, par_a1))

    if not os.path.exists('%s.npz' % fname):
        print par_a1, par_om
        output = Popen(['../jj2_identical.py', '--i1=0.05',
                        '--i2=0.0', '--alpha=0.69:0.92:250', '--d0=0.0', '--deterministic',
                        '--a1=%s' % par_a1, '--omega=%s' % par_om,
                        '--a2=0.0', '--paths=256', '--spp=800',
                        '--samples=800', '--simperiods=2500',
                        '--rng=xs32', '--output_mode=summary',
                        '--output=%s' % fname], stdout=PIPE).communicate()[0]
