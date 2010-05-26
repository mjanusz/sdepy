#!/usr/bin/python

# Assuming a set of (i1, nm) scans is located in profiles/, generate plots for
# them and collect statistics about where NM is the strongest.
#
# ./plot_i1_profiles.py

import sys
import glob
import os
import numpy as np

import matplotlib
matplotlib.use('GTKAgg')

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

from pylab import *

def prep_array(name, init_val):
    out = {}
    out[name] = [init_val, init_val]
    out['%s_name' % name] = ['', '']
    out['noisy_%s' % name] = [init_val, init_val]
    out['noisy_%s_name' % name] = ['', '']
    return out

parser = OptionParser()
parser.add_option('--format', dest='format', type='choice',
        choices=['png', 'pdf'], default='png')
parser.add_option('--figw', dest='figw', type='float', default=8.0)
options, args = parser.parse_args()

intensity = prep_array('min', 0)
extent = prep_array('max', 0)
noise_induced = prep_array('max', -1000)

if options.format == 'pdf':
    rc('savefig', dpi=300.0)
    rc('figure', figsize=[options.figw, 0.75 * options.figw])
    rc('text', usetex=True, fontsize=10)
    rc('axes', labelsize=10)
    rc('legend', fontsize=10)
    rc('xtick', labelsize=8)
    rc('ytick', labelsize=8)

def mmax(arg):
    if len(arg):
        return np.max(arg)
    else:
        return 0

for fname in glob.glob('profiles/*.npz'):
    noisy = fname.replace('.npz', '_d0.0001.npz')
    if not os.path.exists(noisy):
        continue

    det = np.load(fname)
    stoch = np.load(noisy)

    for particle in [0,1]:
        # Intensity
        t = np.min(det['main'][:,particle])

        if t < intensity['min'][particle]:
            intensity['min'][particle] = t
            intensity['min_name'][particle] = fname

        t = np.min(stoch['main'][:,particle])

        if t < intensity['noisy_min'][particle]:
            intensity['noisy_min'][particle] = t
            intensity['noisy_min_name'][particle] = fname

        # Extent
        t = mmax(np.argwhere(det['main'][:,particle] < -0.001))

        if t > extent['max'][particle]:
            extent['max'][particle] = t
            extent['max_name'][particle] = fname

        t = mmax(np.argwhere(stoch['main'][:,particle] < -0.001))

        if t > extent['noisy_max'][particle]:
            extent['noisy_max'][particle] = t
            extent['noisy_max_name'][particle] = fname

        # Noise-induced
        t = np.sum(det['main'][(stoch['main'][:,particle] < -0.001),particle])

        if t > noise_induced['max'][particle]:
            noise_induced['max'][particle] = t
            noise_induced['max_name'][particle] = fname

    cla()
    clf()

    plot(det['i1'], det['main'][:,0], label='$v_1,\, D_0 = 0$')
    plot(det['i1'], det['main'][:,1], label='$v_2,\, D_0 = 0$')
    plot(det['i1'], stoch['main'][:,0], label='$v_1,\, D_0 = 0.0001$')
    plot(det['i1'], stoch['main'][:,1], label='$v_2,\, D_0 = 0.0001$')

    xlim(det['i1'][0], det['i1'][-1])

    ylabel(r'$\langle v \rangle$')
    xlabel(r'$i_1$')
    legend(loc='lower right')
    grid()
    savefig(fname.replace('.npz', '.' + options.format))

print 'Stats:'
print '------'
print 'Intensity:'
print '  %s (%f)' % (intensity['min_name'][0], intensity['min'][0])
print '  %s (%f)' % (intensity['min_name'][1], intensity['min'][1])
print '  %s (%f)' % (intensity['noisy_min_name'][0], intensity['noisy_min'][0])
print '  %s (%f)' % (intensity['noisy_min_name'][1], intensity['noisy_min'][1])
print 'Extent:'
print '  %s (%f)' % (extent['max_name'][0], extent['max'][0])
print '  %s (%f)' % (extent['max_name'][1], extent['max'][1])
print '  %s (%f)' % (extent['noisy_max_name'][0], extent['noisy_max'][0])
print '  %s (%f)' % (extent['noisy_max_name'][1], extent['noisy_max'][1])
print 'Noise-induced:'
print '  %s' % noise_induced['max_name'][0]
print '  %s' % noise_induced['max_name'][1]


