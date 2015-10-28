import sys
import numpy as np

bands = ['G', 'R', 'I', 'Z', 'Y']

catname = sys.argv[1]

f = open(catname, 'r')
lines = f.readlines()
f.close()

n=0
found = False
while not found and n<len(lines):
    if lines[n][0] != '#':
	found = True
	if found:
	    keys = lines[n-1].split('#')[1].split(',')
	    for j in range(0, len(keys)):
		if 'ra2000' in keys[j]:
		    racol = j
		elif 'decl2000' in keys[j]:
		    deccol = j
    else:
	n += 1

outlines = []
outlines.append('#? ra\tdec\tsw\tsh\tfilter\timage\tmask\tvariance\ttype\n')
wgetlines = []

for i in range(n, min(200+n, len(lines)+n)):
    line = lines[i].split(',')
    ra = float(line[racol])
    dec = float(line[deccol])
    for band in bands:
	outlines.append('%9.7f %9.7f 2asec 2asec HSC-%s true true true coadd\n'%(ra, dec, band))
	wgetlines.append('https://hscdata.mtk.nao.ac.jp:4443/das_quarry/cgi-bin/quarryImage?ra=' \
	+ '%9.7f'%ra + '&dec=%9.7f'%dec \
	+ '&sw=2asec&sh=2asec&type=coadd&image=on&filter=HSC-%s'%band \
	#+ '&tract=&rerun=\n')
	+ '\n')

f = open('DAS_Quarry.input', 'w')
f.writelines(outlines)
f.close()

f = open('urllist.txt','w')
f.writelines(wgetlines)
f.close()


