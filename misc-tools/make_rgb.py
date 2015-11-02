import pyfits
import sys
import pylab
import numpy as np

objname = sys.argv[1]

cutout_dir = './'
fig_dir = './'

if len(sys.argv > 1):
    cutout_dir = sys.argv[2]
    if len(sys.argv > 2):
	fig_dir = sys.argv[3]

ifile = cutout_dir+objname+'_i.fits'
rfile = cutout_dir+objname+'_r.fits'
gfile = cutout_dir+objname+'_g.fits'

iflux = pyfits.open(ifile)[0].data.copy()
rflux = pyfits.open(rfile)[0].data.copy()
gflux = pyfits.open(gfile)[0].data.copy()
    
iflux[iflux<0.] = 0.
i99 = np.percentile(iflux, 99.)
iflux *= 255./i99
iflux[iflux>255.] = 255.
iflux = np.uint8(iflux.round())
iflux = np.flipud(iflux)

shape = iflux.shape

rflux[rflux<0.] = 0.
r99 = np.percentile(rflux, 99.)
rflux *= 255./r99
rflux[rflux>255.] = 255.
rflux = np.uint8(rflux.round())
rflux = np.flipud(rflux)

gflux[gflux<0.] = 0.
g99 = np.percentile(gflux, 99.)
gflux *= 255./g99
gflux[gflux>255.] = 255.
gflux = np.uint8(gflux.round())
gflux = np.flipud(gflux)

rgbarray = np.array((iflux.T, rflux.T, gflux.T)).T

dpi=10
pylab.figure(figsize=(shape[0]/float(10*dpi),shape[1]/float(10*dpi)), dpi=dpi)
pylab.axes([0., 0., 1., 1.])
pylab.imshow(rgbarray)
pylab.xticks(())
pylab.yticks(())
pylab.savefig(fig_dir+objname+'_rgb.png', dpi=10*dpi)
pylab.close()

