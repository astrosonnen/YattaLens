import sys
import numpy as np

bands = ['G', 'R', 'I', 'Z', 'Y']

catname = sys.argv[1]

f = open(catname, 'r')
radec = np.loadtxt(f, usecols=(0, 1), delimiter=',')
f.close()

nobj = min(200, radec.shape[0])

outlines = []
outlines.append('#? ra\tdec\tsw\tsh\tfilter\timage\tmask\tvariance\ttype\n')
wgetlines = []

for i in range(0, nobj):
    ra = radec[i, 0]
    dec = radec[i, 1]
    for band in bands:
	outlines.append('%9.7f %9.7f 2asec 2asec HSC-%s true true true coadd\n'%(ra, dec, band))
	wgetlines.append('https://hscdata.mtk.nao.ac.jp:4443/das_quarry/cgi-bin/quarryImage?ra=' \
	+ '%9.7f'%ra + '&dec=%9.7f'%dec \
	+ '&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-%s'%band \
	+ '&tract=&rerun=\n')

f = open('DAS_Quarry.input', 'w')
f.writelines(outlines)
f.close()

f = open('urllist.txt','w')
f.writelines(wgetlines)
f.close()


