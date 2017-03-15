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
from master_server_tools import labeling
import pyfits
from lsst.afw.image import ExposureF
from lsst.pex.exceptions.exceptionsLib import LsstCppException


parent_dir_15b = "/lustre2/HSC_DR/dr1/s15b/data/s15b_wide/" # use this for 15b
butler_15b = lsst.daf.persistence.Butler(parent_dir_15b)
skyMap_15b = butler_15b.get("deepCoadd_skyMap", immediate=True)

parent_dir_16a="/lustre2/HSC_DR/dr1/s16a/data/s16a_wide/"
butler_16a = lsst.daf.persistence.Butler(parent_dir_16a)
skyMap_16a = butler_16a.get("deepCoadd_skyMap", immediate=True)

coadd_dir = 'deepCoadd/'

def extract_posflag(input_image, x, y):
    flag_pos=0
    if os.path.isfile(input_image):
        exposure = ExposureF(input_image)
        psf = exposure.getPsf()
        pos = psf.getAveragePosition()
        pos[0] = int(float(x)+0.5)
        pos[1] = int(float(y)+0.5)
        try:
            psfImageKer = psf.computeKernelImage(pos)
        except LsstCppException:
            print "Cannot compute CoaddPsf for ", input_image, " at the location of the target"
            flag_pos=1

    return flag_pos

###########
def genpsfimage(input_image, flagp, x, y, output):

    if os.path.exists(output):
       print output,"exists already, skipping creation of this image"
    else:
        if os.path.exists(input_image):
            exposure = ExposureF(input_image)
            psf = exposure.getPsf()
            pos = psf.getAveragePosition()
            if(flagp==0):
                pos[0] = int(float(x)+0.5)
                pos[1] = int(float(y)+0.5)
                print "Computing PSF at the location of the target"
            else:
                print "Computing PSF at the center of the patch"

            psfImageKer = psf.computeKernelImage(pos)
            image_psf = psfImageKer.getArray()

            nx1 = psfImageKer.getDimensions()[0]
            ny1 = psfImageKer.getDimensions()[1]
            n_psf = min(nx1, ny1)

            nx = n_psf
            ny = n_psf

            lx = (nx1-nx)/2
            ly = (ny1-ny)/2

            rx=lx+nx
            ry=ly+ny

            hdu = pyfits.open(input_image)
            fitshdu = hdu[1]
            fitshdu.data = image_psf[ly:ry,lx:rx]
            fitshdu.writeto(output, clobber=True)


def getcutout_and_psf(ra, dec, band, name, hsize=50, outdir='/', dr='16a'):

    coord = lsst.afw.coord.IcrsCoord(lsst.afw.geom.Point2D(ra, dec))

    tract, patch = skyMap.findClosestTractPatchList([coord])[0]

    if dr=='16a':
        parent_dir = parent_dir_16a
        butler = butler_16a
    elif dr=='15b':
        parent_dir = parent_dir_15b
        butler = butler_15b
    else:
        antani

    tract1=tract.getId()
    patch1="%d,%d"%(patch[0].getIndex())

    filtname = 'HSC-%s'%band.upper()
    input_image = parent_dir+coadd_dir+filtname+'/'+str(tract1)+'/'+str(patch1)+'/calexp-'+filtname+'-'+str(tract1)+'-'+str(patch1)+'.fits'

    sciname = name+'_%s_sci.fits'%band
    varname = name+'_%s_var.fits'%band
    psfname = name+'_%s_psf.fits'%band

    if os.path.isfile(input_image):# and not os.path.isfile(sciname):
        coadd = butler.get("deepCoadd_calexp", tract=tract.getId(), patch="%d,%d" % patch[0].getIndex(), \
                           filter=filtname)

        pixel = coadd.getWcs().skyToPixel(coord)
        pixel = lsst.afw.geom.Point2I(pixel)
        bbox = lsst.afw.geom.Box2I(pixel, pixel)   # 1-pixel box
        bbox.grow(hsize)
        bbox.clip(coadd.getBBox(lsst.afw.image.PARENT))  # clip to overlap region
        if not bbox.isEmpty():
            subImage = lsst.afw.image.ExposureF(coadd, bbox, lsst.afw.image.PARENT)
            outfits = name+'_'+filtname+'.fits'
            subImage.writeFits(outfits) # makes giant fits file

            call("imcopy %s[1] %s/%s"%(outfits, outdir, sciname),shell=1)
            ## if you want to extract the variance image
            call("imcopy %s[3] %s/%s"%(outfits, outdir, varname),shell=1)
            # removes huge fits file
            os.system('rm %s'%outfits)

            input_dir = parent_dir+coadd_dir+filtname+'/'+str(tract1)+'/'+str(patch1)

            x=pixel[0]
            y=pixel[1]

            flagp=extract_posflag(input_image, x, y)
            genpsfimage(input_image, flagp, x, y, outdir+psfname)

            return True

        else:
            print 'bbox is empty'
            return False
    else:
        print ra, dec, input_image, "not found"
        return False


def fetch_data(radec, outname=None, bands=None, hsize=50, outdir='/'):

    if outname is None:
        outname = labeling.coords2name(radec[0], radec[1])

    if bands is None:
        bands = ('g', 'r', 'i', 'z', 'y')

    # tries to look for data in 16a

    for band in bands:

        found = getcutout_and_psf(radec[0], radec[1], band, outname, hsize=hsize, dr='16a', outdir=outdir)

        if not found:
            found = getcutout_and_psf(radec[0], radec[1], band, outname, hsize=hsize, dr='16a', outdir=outdir)

