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

################
filename=sys.argv[1]
outdir1=sys.argv[2]
boxsize=int(sys.argv[3])
    
#parent_dir="/lustre/Subaru/SSP/rerun/yasuda/SSP3.8.5_20150725/"
#parent_dir="/lustre2/HSC_DR/dr1/s15b/data/s15b_wide/" # use this for 15b
parent_dir="/lustre2/HSC_DR/dr1/s16a/data/s16a_wide/"

coadd_dir = 'deepCoadd/'

butler = lsst.daf.persistence.Butler(parent_dir)
skyMap = butler.get("deepCoadd_skyMap", immediate=True)

inpfilt=['HSC-G','HSC-R','HSC-I','HSC-Y','HSC-Z']
filt=['g','r','i','y','z']
outdir0='fitsdir/'
ra,dec=np.loadtxt(filename,usecols=(0,1),unpack=1)
ra = np.atleast_1d(ra)
dec = np.atleast_1d(dec)

for jj in range(ra.size):
    coord = lsst.afw.coord.IcrsCoord(lsst.afw.geom.Point2D(ra[jj],dec[jj]))
    rrh,ddh=deg2hms(ra[jj],dec[jj])
    for tract, patch in skyMap.findClosestTractPatchList([coord]):
        for ii in range(len(inpfilt)):
            tract1=tract.getId()
            patch1="%d,%d"%(patch[0].getIndex())
            input_image = parent_dir+coadd_dir+inpfilt[ii]+'/'+str(tract1)+'/'+str(patch1)+'/calexp-'+inpfilt[ii]+'-'+str(tract1)+'-'+str(patch1)+'.fits'
            
            if os.path.isfile(input_image) and not os.path.isfile(outdir1+'/'+rrh+ddh+'_'+filt[ii]+'_sci.fits'):
                coadd = butler.get("deepCoadd_calexp", tract=tract.getId(),
                                   patch="%d,%d" % patch[0].getIndex(),
                                   filter=inpfilt[ii])  # your filter here
                #print ra[jj],dec[jj], input_image
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
		# removes huge fits file
		os.system('rm %s'%outfits)
            else:
                print ra[jj],dec[jj], input_image,"not found"
