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


def getcutout_and_psf(ra, dec, band, name, hsize=50, outdir='/', dr='16a'):

    if dr=='16a':
        parent_dir="/lustre2/HSC_DR/dr1/s16a/data/s16a_wide/"

    elif dr=='15b':
        parent_dir="/lustre2/HSC_DR/dr1/s15b/data/s15b_wide/" # use this for 15b

    else:
        antani

    coadd_dir = 'deepCoadd/'

    butler = lsst.daf.persistence.Butler(parent_dir)
    skyMap = butler.get("deepCoadd_skyMap", immediate=True)

    coord = lsst.afw.coord.IcrsCoord(lsst.afw.geom.Point2D(ra, dec))

    tract, patch = skyMap.findClosestTractPatchList([coord])[0]

    tract1=tract.getId()
    patch1="%d,%d"%(patch[0].getIndex())

    filtname = 'HSC-%s'%band.upper()
    input_image = parent_dir+coadd_dir+filtname+'/'+str(tract1)+'/'+str(patch1)+'/calexp-'+filtname+'-'+str(tract1)+'-'+str(patch1)+'.fits'

    sciname = name+'_%s_sci.fits'%band
    varname = name+'_%s_var.fits'%band

    if os.path.isfile(input_image) and not os.path.isfile(sciname):
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

