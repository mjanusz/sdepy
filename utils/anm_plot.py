#!/usr/bin/python

# Plot average velocity using a smart color map designed to highlight
# transport properties of the system.
#
# Arguments:
#  datafile
#  name of the parameter to plot on the X axis
#  name of the parameter to plot on the Y axis
#  [indices for the remining multiple-valued parameter]
#  [index of the innermost dimension if > 1]
#  [output file name]

import sys
import numpy as np

import matplotlib
matplotlib.use('GTKAgg')

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from pylab import *

def make_plot(dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, desc):
    a = abs(min(np.min(dplot[slicearg]), 0.0))
    b = abs(np.max(dplot[slicearg]))
    sc = a / (a+b) / 0.5
    sc2 = b / (a+b) / 0.5

    anm = {
        'red': [
            (0., 0.0, 0.0),
            (sc*0.5, 1.0, 1.0),
            (sc*0.5 + (0.817460-0.5)*sc2, 1.0, 1.0),
            (1.0, 0.8, 0.8)],

        'green': [
            (0., 0.0, 0.0),
            (sc*0.4, 1.0, 1.0),
            (sc*0.5, 1.0, 1.0),
            (sc*0.5 + (0.626984-0.5)*sc2, 1.0, 1.0),
            (sc*0.5 + (0.817460-0.5)*sc2, 0.6, 0.6),
            (1.0, 0.0, 0.0)],

        'blue': [
            (0.0, 0.4, 0.4),
            (sc*0.25, 1.0, 1.0),
            (sc*0.5, 1.0, 1.0),
            (sc*0.5 + (0.626984-0.5)*sc2, 0., 0.),
            (1.0, 0.0, 0.0)]
        }

    if sc == 0.0:
        for k in anm.iterkeys():
            anm[k] = anm[k][1:]
    # Fix color scale if only positive values are present.
    elif a < 0.001:
        for k in anm.iterkeys():
            anm[k][0] = (0., 1.0, 1.0)

    anm_cmap = LinearSegmentedColormap('ANM', anm)
    plt.register_cmap(cmap=anm_cmap)

    aspect = ((data[pars[xidx]][-1] - data[pars[xidx]][0]) /
              (data[pars[yidx]][-1] - data[pars[yidx]][0])) / 1.5

    imshow(dplot[slicearg],
           extent=(data[pars[xidx]][0], data[pars[xidx]][-1],
                   data[pars[yidx]][0], data[pars[yidx]][-1]),
           aspect=aspect,
           interpolation='bilinear',
#       interpolation='nearest',
           cmap = anm_cmap,
           origin='lower')
    ylabel(yvar)
    xlabel(xvar)
    title(desc)
    colorbar()

if __name__ == '__main__':
    fname = sys.argv[1]
    xvar = sys.argv[2]
    yvar = sys.argv[3]

    data = np.load(fname)
    pars = list(data['par_multi']) + list(data['scan_vars'])

    print 'Parameters are (in order): ', pars
    print 'Shape: ', data['main'].shape

    idxs = set(range(len(pars)))
    xidx = pars.index(xvar)
    yidx = pars.index(yvar)
    idxs.remove(xidx)
    idxs.remove(yidx)

    # Now scan the command line arguments for info about
    # which values to use for the remanining parameters
    argidx = 4
    slicearg = []
    desc = ''

    for i in range(len(pars)):
        if i in idxs:
            slicearg.append(int(sys.argv[argidx]))
            argidx += 1
            print '%s = %s' % (pars[i], data[pars[i]][slicearg[-1]])
            desc += '%s=%s ' % (pars[i], data[pars[i]][slicearg[-1]])
        else:
            slicearg.append(slice(None))

    if data['main'].shape[-1] > 1:
        slicearg.append(sys.argv[argidx])
        argidx += 1
    else:
        slicearg.append(0)

    dplot = data['main']

    if xidx < yidx:
        dplot = np.swapaxes(dplot, xidx, yidx)

    print 'min/max: ', np.min(dplot[slicearg]), np.max(dplot[slicearg])

    make_plot(dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, desc)

    if len(sys.argv) > argidx:
        savefig(sys.argv[argidx])
    else:
        ion()
        show()
