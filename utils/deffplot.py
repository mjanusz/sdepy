#!/usr/bin/python

# Plot D_eff(force) for multiple values of noise and a single value 
# of gamma (see bistable.py).
# 
# Illustrates basic data processing and matplotlib use.
#
# Sample usage:
# ./deffplot.py 0.5 iv_0

import numpy as np
import glob
import sys
from collections import deque
from StringIO import StringIO

import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

from matplotlib.font_manager import fontManager, FontProperties 
font = FontProperties(size='x-small')

from pylab import *
ion()

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

parser = OptionParser()
parser.add_option('--offset', dest='offset', 
        help='time offset from the end (expressed in time steps)', type='int', default=0)
parser.add_option('--output', dest='output',
        help='output file', type='string', default=None)
options, args = parser.parse_args()

gamma = args[0]     # gamma value
files = []
postfixes = []

if len(args) > 1:
    for i in range(1, len(args)):
        postfixes.append(args[i])
else:
    postfixes.append('')

for postfix in postfixes:
    files.extend(glob.glob('g%s_d[0-9].???%s' % (gamma, postfix)))

def find_bs(data):
    return np.sum(data[:][:,0] == data[-1:][:,0])

def find_bs_f(f):
    with open(f) as fp:
        cnt = 0
        for line in fp:
            if line[0] == '\n':
                break
            if line[0] != '#':
                cnt += 1

    return cnt

for file in files:
    print file, 
    bs = find_bs_f(file)

    # No need to load the whole file, which is slooow.
    # dat = np.loadtxt(file)
    # bs = find_bs(dat)
    
    if options.offset == 0:
        dat = np.loadtxt(StringIO(''.join(deque(open(file), 2*bs))))
        semilogy(dat[-bs:][:,1], dat[-bs:][:,2],
                '.-', label='%s, t_max=%.4e' % (file, dat[-1:][:,0]))
    else:
        dat = np.loadtxt(StringIO(''.join(deque(open(file), (options.offset + 10)*(bs+1)))))
        semilogy(dat[-bs*(options.offset+1):-bs*options.offset][:,1], 
                 dat[-bs*(options.offset+1):-bs*options.offset][:,2],
                '.-', label='%s, t_max=%.4e' % (file, dat[-bs*options.offset,0]))
       

grid(True)
legend(prop=font, loc='lower right')

if options.output:
    savefig(options.output, format='pdf')
else:
    show()
