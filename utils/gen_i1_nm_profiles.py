#!/usr/bin/python

# Given a list of files with scan data for a constant value of alpha, find areas
# of the parameter space where NM has been detected for multiple values of i1
# or noise, and generate (i1, <v>) profiles for them.  The (omega, a1) parameter
# ranges in files supplied as arguments to this script have to match exactly.
#
# ./gen_i1_nm_profiles.py scan1.npz scan2.npz ...

from subprocess import Popen, PIPE
import os
import numpy as np
import sys

a = np.load(sys.argv[1])
omega = a['omega']
a1 = a['a1']
particle = 1

neg_mob_map = (a['main'][:,:,particle] < -0.001)

for file in sys.argv[2:]:
    a = np.load(file)
    neg_mob_map = np.logical_and(neg_mob_map, a['main'][:,:,particle] < -0.001)

yi, xi = np.nonzero(neg_mob_map)

for i in range(0, len(xi)):
    x = xi[i]
    y = yi[i]
    par_a1 = a1[x]
    par_om = omega[y]

    fname = os.path.join('profiles',
                         'jj2_ident_anm_prof_omega%s_a1%s_d0.0001' % (par_om, par_a1))

    if not os.path.exists('%s.npz' % fname):
        print par_a1, par_om, a['main'][y,x,particle]
        output = Popen(['../jj2_identical.py', '--i1=0.0:0.1:250',
                        '--i2=0.0', '--alpha=0.77', '--d0=0.0001',
                        '--a1=%s' % par_a1, '--omega=%s' % par_om,
                        '--a2=0.0', '--paths=512', '--spp=800',
                        '--samples=800', '--simperiods=2500',
                        '--rng=xs32', '--output_mode=summary',
                        '--output=%s' % fname], stdout=PIPE).communicate()[0]
