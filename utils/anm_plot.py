#!/usr/bin/python

import sys
import numpy as np

import matplotlib
matplotlib.use('GTKAgg')

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from pylab import *

fname = sys.argv[1]
xvar = sys.argv[2]
yvar = sys.argv[3]

data = np.load(fname)
pars = list(data['par_multi']) + list(data['scan_vars'])

print 'Parameters are (in order): ', pars

idxs = set(range(len(pars)))
xidx = pars.index(xvar)
yidx = pars.index(yvar)
idxs.remove(xidx)
idxs.remove(yidx)

# Now scan the command line arguments for info about
# which values to use for the remanining parameters
argidx = 4
slicearg = []

for i in range(len(pars)):
    if i in idxs:
        slicearg.append(int(sys.argv[argidx]))
        argidx += 1
        print '%s = %s' % (pars[i], data[pars[i]][slicearg[-1]])
    else:
        slicearg.append(slice(None))

if data['main'].shape[-1] > 1:
    slicearg.append(sys.argv[argidx])
else:
    slicearg.append(0)

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }

cdict2 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 1.0),
                   (1.0, 0.1, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.1),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.0, 0.0))
        }

invjet = {'red':   ((0., 0, 0),
                    (0.35, 0, 0),
                    (0.49, 0, 0.5),
                    (0.69,1, 1),
                    (1, 1, 1)),
          'green': ((0., 0, 0),
                    (0.125,0, 0),
                    (0.375,1, 1),
                    (0.49, 1, 0),
                    (0.61,0,0),
                    (1, 1, 1)),
          'blue':  ((0., 0.5, 0.5),
                    (0.11, 1, 1),
                    (0.49, 1, 0),
                    (0.65,0, 0),
                    (1, 0, 0))}

pm3d = {'red': ((0., 0, 0),
                (0.25, 0.7, 0.7),
                (0.5, 0.4, 0.6),
                (1.0, 1.0, 1.0)),
        'green': ((0., 0, 0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)),
        'blue': ((0., 0.0, 0.0),
                (0.25, 0.8, 0.8),
                (0.5, 0.0, 0),
                (1.0, 0, 0))
        }

anm = {
    'red': (
        (0.0, 0, 0),
        (0.5, 1, 1),
        (0.75, 1, 1),
        (1, 1, 1)),
    'green': (
        (0.0, 0, 0),
        (0.48, 1, 1),
        (0.5, 1, 1),
        (0.75, 0.5, 0.5),
        (1, 1, 1)),

    'blue': (
        (0.0, 0.5, 0.5),
        (0.5, 1, 1),
        (1, 0, 0))
    }

anm2 = {
    'red': (
        (0., 0.0, 0.0),
        (0.5, 1.0, 1.0),
        (0.817460, 1.0, 1.0),
        (1.0, 0.8, 0.8)),

    'green': (
        (0., 0.0, 0.0),
        (0.4, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.626984, 1.0, 1.0),
        (0.817460, 0.6, 0.6),
        (1.0, 0.0, 0.0)),

    'blue': (
        (0.0, 0.4, 0.4),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.626984, 0., 0.),
        (1.0, 0.0, 0.0))
    }

blue_red2 = LinearSegmentedColormap('BlueRed2', anm2)
plt.register_cmap(cmap=blue_red2)

dplot = data['main']

if xidx < yidx:
    dplot = np.swapaxes(dplot, xidx, yidx)

print 'min/max: ', np.min(dplot[slicearg]), np.max(dplot[slicearg])

imshow(dplot[slicearg],
       extent=(data[pars[xidx]][0], data[pars[xidx]][-1],
               data[pars[yidx]][0], data[pars[yidx]][-1]),
       vmin = -1.0,
       vmax = 1.0,
       aspect = float(len(data[pars[xidx]])) / len(data[pars[yidx]]) / 2,
#       interpolation='sinc',
       interpolation='nearest',
#       cmap=get_cmap('RdBu'),
       cmap = blue_red2,
       origin='lower')
ylabel(yvar)
xlabel(xvar)
colorbar()

#if len(sys.argv) > 3:
#    savefig(sys.argv[3])
#else:
ion()
show()
