#!/usr/bin/env python
import lsst.daf.persistence
import lsst.afw.coord
import lsst.afw.image
import lsst.afw.geom
from math import *
import numpy as np
from subprocess import call
import os
import sys
import pyfits

from lsst.afw.image import ExposureF
from lsst.pex.exceptions.exceptionsLib import LsstCppException


inpfilt=['HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y']
filt=['g','r','i','z','y']
outdir0='fitsdir/'

if not os.path.isdir(outdir0):
    os.system('mkdir %s'%outdir0)

def deg2hms(rr,dd):
    flag=0
    rr1=floor(rr/15.)
    rr2=floor((rr/15.-rr1)*60)
    rr3=(((rr/15.-rr1)*60) - rr2)*60

    if(dd<0):
        flag=1
        dd=-1*dd

    dd1=floor(dd)
    dd2=floor((dd-dd1)*60)
    dd3=(((dd-dd1)*60)-dd2)*60

    if(flag):
        return "%02d%02d%02d"%(rr1,rr2,floor(rr3)), "-%02d%02d%02d"%(dd1,dd2,floor(dd3))
    else:
        return "%02d%02d%02d"%(rr1,rr2,floor(rr3)), "+%02d%02d%02d"%(dd1,dd2,floor(dd3))

###########

def genpsfimage(input_image, xy, output):

    flag_pos=0

    x, y = xy
    if os.path.isfile(input_image):
        exposure = ExposureF(input_image)
        psf = exposure.getPsf()
        pos = (int(float(x)+0.5), int(float(y)+0.5))

        try:
            psfImageKer = psf.computeKernelImage(pos)
        except LsstCppException:
            print "Cannot compute CoaddPsf for ", filename, " at the location of the target"
            print "Computing PSF at the center of the patch"
            pos = psf.getAveragePosition()
            psfImageKer = psf.computeKernelImage(pos)
            break

        image_psf = psfImageKer.getArray()

        nx1 = psfImageKer.getDimensions()[0]
        ny1 = psfImageKer.getDimensions()[1]
        n_psf = min(nx1,ny1)

        nx = n_psf
        ny = n_psf

        lx = (nx1-nx)/2
        ly = (ny1-ny)/2

        rx=lx+nx
        ry=ly+ny

        hdu = pyfits.open(input_images[ii])
        fitshdu = hdu[1]
        fitshdu.data = image_psf[ly:ry,lx:rx]
        fitshdu.writeto(output)

    else:
        continue

################
filename=sys.argv[1]
outdir1=sys.argv[2]
boxsize=int(sys.argv[3])

f = open(filename, 'r')
lines = f.readlines()
f.close()

nobj = len(lines)

ra, dec= np.loadtxt(filename, usecols=(0, 1), unpack=1)
ra = np.atleast_1d(ra)
dec = np.atleast_1d(dec)

# sorts the catalog by RA
zipped = zip(ra, lines, dec)
zipped.sort()
sra, slines, sdec = zip(*zipped)

catalog = []
for i in range(nobj):
    hra, ddec = deg2hms(sra[i], sdec[i])
    name = hra+ddec
    line = slines[i].split()
    extra = ''
    for field in line[2:]:
        extra += ' %s'%field

    dic = {'name': name, 'ra': sra[i], 'dec': sdec[i], 'extra': extra}
    for band in filt:
        dic[band] = None

    catalog.append(dic)

parent_dirs=["/lustre2/HSC_DR/dr1/s16a/data/s16a_wide/", "/lustre2/HSC_DR/dr1/s15b/data/s15b_wide/"]
drs = ['16a', '15b']

coadd_dir = 'deepCoadd/'


for n in range(2):
    dr = drs[n]
    parent_dir = parent_dirs[n]
    butler = lsst.daf.persistence.Butler(parent_dir)
    skyMap = butler.get("deepCoadd_skyMap", immediate=True)

    for jj in range(nobj):
        coord = lsst.afw.coord.IcrsCoord(lsst.afw.geom.Point2D(catalog[jj]['ra'], catalog[jj]['dec']))
        rrh, ddh = deg2hms(catalog[jj]['ra'], catalog[jj]['dec'])

        for tract, patch in skyMap.findClosestTractPatchList([coord]):
            for ii in range(len(inpfilt)):
                tract1=tract.getId()
                patch1="%d,%d"%(patch[0].getIndex())
                input_image = parent_dir+coadd_dir+inpfilt[ii]+'/'+str(tract1)+'/'+str(patch1)+'/calexp-'+inpfilt[ii]+'-'+str(tract1)+'-'+str(patch1)+'.fits'

                if os.path.isfile(input_image) and not os.path.isfile(outdir1+'/'+rrh+ddh+'_'+filt[ii]+'_sci.fits'):
                    coadd = butler.get("deepCoadd_calexp", tract=tract.getId(),
                                       patch="%d,%d" % patch[0].getIndex(),
                                       filter=inpfilt[ii])  # your filter here

                    pixel = coadd.getWcs().skyToPixel(coord)
                    pixel = lsst.afw.geom.Point2I(pixel)

                    bbox = lsst.afw.geom.Box2I(pixel, pixel)   # 1-pixel box
                    bbox.grow(boxsize)    # now a 31x31 pixel box
                    bbox.clip(coadd.getBBox(lsst.afw.image.PARENT))  # clip to overlap region

                    if bbox.isEmpty():
                        continue

                    subImage = lsst.afw.image.ExposureF(coadd, bbox, lsst.afw.image.PARENT)
                    outfits=outdir0+rrh+ddh+'_'+inpfilt[ii]+'.fits'
                    outfits1=rrh+ddh+'_'+filt[ii]+'_sci.fits'
                    subImage.writeFits(outfits)
                    call("imcopy %s[1] %s/%s"%(outfits,outdir1,outfits1),shell=1)
                    ## if you want to extract the variance image
                    outfits2=rrh+ddh+'_'+filt[ii]+'_var.fits'
                    call("imcopy %s[3] %s/%s"%(outfits,outdir1,outfits2),shell=1)

                    boxshape = bbox.getDimensions()

                    catalog[jj][filt[ii]] = '%s-%d-%d'%(dr, boxshape[0], boxshape[1])

                    x=pixel[0]
                    y=pixel[1]

                    out_psffits = rrh+ddh+'_'+filt[ii]+'_psf.fits'

                    genpsfimage(input_image, (x, y), out_psffits)

                    # removes huge fits file
                    os.system('rm %s'%outfits)

                else:
                    print ra[jj],dec[jj], input_image,"not found"

outlines = []
for object in catalog:
    outline = '%s %f %f'%(object['name'], object['ra'], object['dec'])
    for band in filt:
        outline += ' %s'%object[band]
    outline += '%s\n'%object['extra']
    outlines.append(outline)

f = open('summary.txt', 'w')
f.writelines(outlines)
f.close()

