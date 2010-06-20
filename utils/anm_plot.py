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

var_disp = {
    'a1': r'$a_1$',
    'a2': r'$a_2$',
    'omega': r'$\omega$',
    'amp': r'$a$',
    'gam': r'$\gamma$',
    'gam1': r'$\gamma_1$',
    'gam2': r'$\gamma_2$',
    'b': r'$b$',
}

def var_display(name):
    global var_disp

    if name in var_disp:
        return var_disp[name]
    else:
        return name

def make_subplot(options, ax, dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, a, b):
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
              (data[pars[yidx]][-1] - data[pars[yidx]][0])) / options.panel_aspect

    im = ax.imshow(dplot[slicearg],
           extent=(data[pars[xidx]][0], data[pars[xidx]][-1],
                   data[pars[yidx]][0], data[pars[yidx]][-1]),
           aspect=aspect,
           interpolation='bilinear',
#           interpolation='nearest',
           vmin = -a,
           vmax = b,
           cmap = anm_cmap,
           origin='lower')

    return im

def make_plot(dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, desc):
    global options

    a = abs(min(np.nanmin(dplot[slicearg]), 0.0))
    b = abs(np.nanmax(dplot[slicearg]))

    cla()
    clf()
    axes([0.1,-0.05,0.95,1.0])
    im = make_subplot(options, gca(), dplot, slicearg, data, pars, xvar, yvar, xidx, yidx, a, b)
    ylabel(var_display(yvar))
    xlabel(var_display(xvar))

    if options.format != 'pdf':
        title(desc)
    colorbar(im, shrink=0.75)

def multi_plot(data, pars, xvar, yvar, xidx, yidx, argidx, slicearg, desc,
        postfix, args, pretend=False):
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
                        desc2, pfx, args, pretend=pretend)
        else:
            sa = slicearg + [int(args[argidx])]
            print '%s = %s' % (pars[iidx], data[pars[iidx]][sa[-1]])
            desc2 = desc + ' %s=%s ' % (pars[iidx], data[pars[iidx]][sa[-1]])
            pfx = postfix + '_%s%03d' % (pars[iidx], sa[-1])
            return multi_plot(data, pars, xvar, yvar, xidx, yidx, argidx+1, sa, desc2,
                    pfx, args, pretend=pretend)
    else:
        if iidx <= xidx:
            slicearg.append(slice(None))
        if iidx <= yidx:
            slicearg.append(slice(None))

        def do_plot(pfix, args, argidx):
            global options

            is_abs = options.abs

            try:
                if args[argidx] == 'abs':
                    argidx += 1
                    is_abs = True

            except IndexError:
                pass

            if is_abs:
                dplot = data['abs']
                dplot[np.isnan(dplot)] = 0.0
            else:
                dplot = data['main']

            if xidx < yidx:
                dplot = np.swapaxes(dplot, xidx, yidx)

            print 'min/max/nans?: ', np.nanmin(dplot[slicearg]), np.nanmax(dplot[slicearg]), np.any(np.isnan(dplot[slicearg]))
            if pretend:
                return (argidx, (dplot, slicearg, data, pars, xvar, yvar, xidx,
                    yidx))
            else:
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
                    do_plot(postfix + '_in%d' % i, args, argidx)
            else:
                slicearg.append(int(args[argidx]))
                argidx += 1
                return do_plot(postfix, args, argidx)
        else:
            slicearg.append(0)
            return do_plot(postfix, args, argidx)

def parse_opts():
    global options
    parser = OptionParser()
    parser.add_option('--format', dest='format', type='choice',
            choices=['png', 'pdf'], default='png')
    parser.add_option('--figw', dest='figw', type='float', default=8.0)
    parser.add_option('--abs', dest='abs', action='store_true', default=False)
    parser.add_option('--output', dest='output', type='string', default='out')
    parser.add_option('--fig_aspect', dest='figaspect', type='float',
            default=0.75)
    parser.add_option('--panel_aspect', dest='panel_aspect', type='float',
            default=1.5)
    parser.add_option('--panels', dest='panels', type='int', default=1)

    options, args = parser.parse_args()

    return options, args

def set_opts():
    global options
    if options.format == 'pdf':
        rc('savefig', dpi=300.0)
        rc('figure', figsize=[options.figw, options.figaspect * options.figw])
        rc('text', usetex=True, fontsize=10)
        rc('axes', labelsize=10)
        rc('legend', fontsize=10)
        rc('xtick', labelsize=8)
        rc('ytick', labelsize=8)

if __name__ == '__main__':
    options, args = parse_opts()
    set_opts()
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

