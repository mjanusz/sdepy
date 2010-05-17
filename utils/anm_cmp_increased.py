#!/usr/bin/python
#
# The 2nd file is where we look for increased ANM.
#
import numpy as np
import sys

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
inneridx = int(sys.argv[3])

dat1 = a['main']
dat2 = b['main']

slicearg = (len(dat1.shape) - 1) * [slice(None)] + [inneridx]
mask = (dat1[slicearg] > dat2[slicearg])
dat2[mask] = 0.0

print 'Found %d sites where ANM is increased.' % np.sum(np.logical_not(mask))

newdict = dict(a.items())
newdict['main'] = dat2

np.savez(sys.argv[4], **newdict)

