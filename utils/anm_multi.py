#!/usr/bin/python

import numpy as np
import anm_plot
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
from pylab import *

nr = 2
nc = 2

options, args = anm_plot.parse_opts()

# Spacing and size of the colorbar as fractions of the figure size.
space = 0.01
cbar_fraction = 0.04
effective_w = 1.0 - cbar_fraction - space

# The inset apsect if the controlling variable and the total figure apsect is
# automatically adjusted.
options.fig_aspect = effective_w * (1.0/options.inset_aspect)

fig = figure()
cax = fig.add_axes([1.0 - cbar_fraction, 0.0, cbar_fraction, 1.0])

w = effective_w/2.0 - space/2.0
h = 0.5 - space/2.0

axs = []

a = 999999
b = -999999
anm_args = []

labels = ['$(a)$', '$(b)$', '$(c)$', '$(d)$']
i = 0

for y in reversed(range(nc)):
    for x in range(nr):
        pos = [x*(effective_w/2.0 + space/2.0), y*(0.5 + space/2.0), w, h]
        a = fig.add_axes(pos)
        if y > 0:
            a.set_xticklabels([])
        if x > 0:
            a.set_yticklabels([])

        t = fig.text(pos[0] + 0.05, pos[1] + h - 0.075, labels[i], horizontalalignment='left') #fontproperties=FontProperties(size=16))
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
        a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
        axs.append(a)
        i += 1

for ax in axs:
    fname = args[0]

    data = np.load(fname)
    pars = list(data['par_multi']) + list(data['scan_vars'])

    xvar = args[1]
    yvar = args[2]

    idxs = set(range(len(pars)))
    xidx = pars.index(xvar)
    yidx = pars.index(yvar)
    idxs.remove(xidx)
    idxs.remove(yidx)

    slicearg = []
    desc = ''

    i, anm_plot_args = anm_plot.multi_plot(
            data, pars, xvar, yvar, xidx, yidx, 3, slicearg, desc, '',
            args, pretend=True)

    anm_args.append(anm_plot_args)
    (dplot, slicearg, data, pars, xvar, yvar, xidx, yidx) = anm_plot_args

    a = min(a, abs(min(np.nanmin(dplot[slicearg]), 0.0)))
    b = max(b, abs(np.nanmax(dplot[slicearg])))

    args = args[i:]

i = 0
for y in reversed(range(nc)):
    for x in range(nr):
        ax = axs[i]
        if x == 0:
            ax.set_ylabel(anm_plot.var_display(yvar))
        if y == 0:
            ax.set_xlabel(anm_plot.var_display(xvar))


        i += 1


for i, ax in enumerate(axs):
    im = anm_plot.make_subplot(options, ax, *anm_args[i], a=a, b=b)


fig.colorbar(im, cax)

#grid.axes_llc.set_xlim([0, 2])
#grid.axes_llc.set_ylim([0.05, 0.25])

#plt.colorbar(im, cax = grid.cbar_axes[0])
#grid.cbar_axes[0].colorbar(im)
#ion()
#show()
plt.savefig(options.output + '.' + options.format, bbox_inches='tight')
