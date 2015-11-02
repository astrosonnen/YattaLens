#this code takes a catalog as produced from an SQL query on the NAOJ website and makes it into an RA-dec-ID list to be used in Anu's script to get images and PSFs.

#currently puts a limit on the number of objects that are actually put in the list.

import numpy as np
import sys

Nobj = 100

racol = 0
deccol = 1
zcol = 8
idcol = 9

f = open(sys.argv[1], 'r')
radecz = np.loadtxt(f, usecols=(racol, deccol, zcol), delimiter=',')
f.close()

ra = radecz[:, 0]
dec = radecz[:, 1]
z = radecz[:, 2]

f = open(sys.argv[1], 'r')
idcol = np.loadtxt(f, usecols=(idcol, ), dtype=int, delimiter=',')
f.close()

outlines = []

count = 0
i = 0

while count < Nobj and i < len(ra):
    if z[i] < 1.:
	outlines.append('%9.7f %9.7f %d\n'%(ra[i], dec[i], idcol[i]))    
	count += 1
    i += 1

outname = 'radec_list.txt'
if len(sys.argv) > 2:
    outname = sys.argv[2]

f = open(outname, 'w')
f.writelines(outlines)
f.close()

