#!/usr/bin/python

# Plot average velocity using a smart color map designed to highlight
# transport properties of the system.
#
# Arguments:
#  datafile
#  name of the parameter to plot on the X axis
#  name of the parameter to plot on the Y axis
#  [indices for the remaining multiple-valued parameter]
#  [index of the innermost dimension if > 1]
#  [output file name]

import sys
import numpy as np
import math

import matplotlib
matplotlib.use('GTKAgg')

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

from pylab import *

parser = OptionParser()
parser.add_option('--format', dest='format', type='choice',
        choices=['png', 'pdf'], default='png')
parser.add_option('--figw', dest='figw', type='float', default=8.0)
options, args = parser.parse_args()

var_disp = {
    'a1': r'$a_1$',
    'a2': r'$a_2$',
    'omega': r'$\omega$',
    'amp': r'$a$',
    'gam': r'$\gamma$',
}

def var_display(name):
    global var_disp

    if name in var_disp:
        return var_disp[name]
    else:
        return name

def make_plot(dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, desc):
    global options

    a = abs(min(np.nanmin(dplot[slicearg]), 0.0))
    b = abs(np.nanmax(dplot[slicearg]))

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

    if b < 0.001:
        anm['red'] = anm['red'][0:1] + [(1, 1, 1)]
        anm['green'] = anm['green'][0:2] + [(1, 1, 1)]
        anm['blue'] = anm['blue'][0:2] + [(1, 1 ,1)]

    anm_cmap = LinearSegmentedColormap('ANM', anm)
    plt.register_cmap(cmap=anm_cmap)

    aspect = ((data[pars[xidx]][-1] - data[pars[xidx]][0]) /
              (data[pars[yidx]][-1] - data[pars[yidx]][0])) / 1.5

    cla()
    clf()
    axes([0.1,-0.05,0.95,1.0])
    imshow(dplot[slicearg],
           extent=(data[pars[xidx]][0], data[pars[xidx]][-1],
                   data[pars[yidx]][0], data[pars[yidx]][-1]),
           aspect=aspect,
           interpolation='bilinear',
#       interpolation='nearest',
           cmap = anm_cmap,
           origin='lower')
    ylabel(var_display(yvar))
    xlabel(var_display(xvar))

    if options.format != 'pdf':
        title(desc)
    colorbar(shrink=0.75)

def multi_plot(data, pars, xvar, yvar, xidx, yidx, argidx, slicearg, desc,
        postfix, args):
    # Scan the command line arguments for info about
    # which values to use for the remaining parameters.
    iidx = argidx - 3

    if iidx < len(pars) - 2:
        if iidx == xidx:
            # This parameter is assigned to one of the axes.
            slicearg.append(slice(None))
            iidx += 1
        elif iidx > xidx:
            iidx += 1

        if iidx == yidx:
            # This parameter is assigned to one of the axes.
            slicearg.append(slice(None))
            iidx += 1
        elif iidx > yidx:
            iidx += 1

        if args[argidx] == '-':
            for val in range(len(data[pars[iidx]])):
                sa = slicearg + [val]
                print '%s = %s' % (pars[iidx], data[pars[iidx]][sa[-1]])
                desc2 = desc + ' %s=%s ' % (pars[iidx], data[pars[iidx]][sa[-1]])
                pfx = postfix + '_%s%03d' % (pars[iidx], sa[-1])
                multi_plot(data, pars, xvar, yvar, xidx, yidx, argidx+1, sa,
                        desc2, pfx, args)
        else:
            sa = slicearg + [int(args[argidx])]
            print '%s = %s' % (pars[iidx], data[pars[iidx]][sa[-1]])
            desc2 = desc + ' %s=%s ' % (pars[iidx], data[pars[iidx]][sa[-1]])
            pfx = postfix + '_%s%03d' % (pars[iidx], sa[-1])
            multi_plot(data, pars, xvar, yvar, xidx, yidx, argidx+1, sa, desc2,
                    pfx, args)
    else:
        if iidx <= xidx:
            slicearg.append(slice(None))
        if iidx <= yidx:
            slicearg.append(slice(None))

        def do_plot(pfix):
            dplot = data['main']
            if xidx < yidx:
                dplot = np.swapaxes(dplot, xidx, yidx)

            print 'min/max/nans?: ', np.nanmin(dplot[slicearg]), np.nanmax(dplot[slicearg]), np.any(np.isnan(dplot[slicearg]))
            make_plot(dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, desc)

            global options

            if len(args) > argidx:
                savefig(args[argidx] + pfix + '.' + options.format, bbox_inches='tight')
            else:
                ion()
                show()

        # More than 1 value on the innermost axis?
        if data['main'].shape[-1] > 1:
            sorig = slicearg
            if args[argidx] == '-':
                argidx += 1
                for i in range(data['main'].shape[-1]):
                    slicearg = sorig + [i]
                    do_plot(postfix + '_in%d' % i)
            else:
                slicearg.append(int(args[argidx]))
                argidx += 1
                do_plot(postfix)
        else:
            slicearg.append(0)
            do_plot(postfix)

if options.format == 'pdf':
    rc('savefig', dpi=300.0)
    rc('figure', figsize=[options.figw, 0.75 * options.figw])
    rc('text', usetex=True, fontsize=10)
    rc('axes', labelsize=10)
    rc('legend', fontsize=10)
    rc('xtick', labelsize=8)
    rc('ytick', labelsize=8)

if __name__ == '__main__':
    fname = args[0]
    data = np.load(fname)
    pars = list(data['par_multi']) + list(data['scan_vars'])

    print 'Parameters are (in order): ', pars
    print 'Shape: ', data['main'].shape

    xvar = args[1]
    yvar = args[2]

    idxs = set(range(len(pars)))
    xidx = pars.index(xvar)
    yidx = pars.index(yvar)
    idxs.remove(xidx)
    idxs.remove(yidx)

    argidx = 3
    slicearg = []
    desc = ''

    multi_plot(data, pars, xvar, yvar, xidx, yidx, argidx, slicearg, desc, '',
            args)

