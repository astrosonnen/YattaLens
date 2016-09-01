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
import struct
import pyfits

from lsst.afw.image import ExposureF
from lsst.pex.exceptions.exceptionsLib import LsstCppException

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
def extract_posflag(input_images):           
    flag_pos=0 
    for filename in (input_images): 
        if os.path.isfile(filename): 
            exposure = ExposureF(filename)
            psf = exposure.getPsf()
            pos = psf.getAveragePosition()
            pos[0] = int(float(x)+0.5)
            pos[1] = int(float(y)+0.5)
            try:
                psfImageKer = psf.computeKernelImage(pos)
            except LsstCppException:
                print "Cannot compute CoaddPsf for ", filename, " at the location of the target"
                flag_pos=1
                break
        else: 
            continue
    return flag_pos

###########
def genpsfimage(input_images,flagp,output):            
    for ii in range(len(input_images)): 
        if os.path.exists(output[ii]):
           print output[ii],"exists already, skipping creation of this image"
           continue
        else: 
            if os.path.exists(input_images[ii]): 
                exposure = ExposureF(input_images[ii])
                psf = exposure.getPsf()
                pos = psf.getAveragePosition()
                if(flagp==0):
                    pos[0] = int(float(x)+0.5)
                    pos[1] = int(float(y)+0.5)
                    if(ii==0):
                        print "Computing PSF at the location of the target"
                else:
                    if(ii==0):
                        print "Computing PSF at the center of the patch"
             
                psfImageKer = psf.computeKernelImage(pos)
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
                fitshdu.writeto(output[ii])
                
            else:
                continue

################
filename=sys.argv[1]
outdir1=sys.argv[2]

parent_dir="/lustre2/HSC_DR/dr1/s15b/data/s15b_wide/"
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
        input_images=[]
        out_psffits=[]
        flag_bound=0 
        for ii in range(len(inpfilt)):
            tract1=tract.getId()
            patch1="%d,%d"%(patch[0].getIndex())
            input_image = parent_dir+coadd_dir+inpfilt[ii]+'/'+str(tract1)+'/'+str(patch1)+'/calexp-'+inpfilt[ii]+'-'+str(tract1)+'-'+patch1+'.fits'
            input_images.append(input_image)
            out_psffits.append(outdir1+'/'+rrh+ddh+'_'+filt[ii]+'_psf.fits')
        for ii in range(len(inpfilt)):
            tract1=tract.getId()
            patch1="%d,%d"%(patch[0].getIndex())
            input_image = parent_dir+coadd_dir+inpfilt[ii]+'/'+str(tract1)+'/'+str(patch1)
             
            if os.path.isdir(input_image) and not os.path.isfile(outdir1+'/'+rrh+ddh+'_'+filt[ii]+'_psf.fits'): 
                coadd = butler.get("deepCoadd_calexp", tract=tract.getId(),
                                   patch="%d,%d" % patch[0].getIndex(),
                                   filter=inpfilt[ii])  # your filter here
                print ra[jj],dec[jj], input_image
                pixel = coadd.getWcs().skyToPixel(coord)
                pixel = lsst.afw.geom.Point2I(pixel)

                x=pixel[0]
                y=pixel[1]

                flagp=extract_posflag(input_images)
                genpsfimage(input_images,flagp,out_psffits)
                break
            else:
                print ra[jj],dec[jj], input_image,"not found"
