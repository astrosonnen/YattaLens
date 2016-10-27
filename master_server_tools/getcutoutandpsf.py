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
    for filt in inpfilt:
        dic[filt] = None

    catalog.append(dic)

parent_dirs=["/lustre2/HSC_DR/dr1/s15b/data/s15b_wide/", "/lustre2/HSC_DR/dr1/s16a/data/s16a_wide/"]
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
                input_image = parent_dir+coadd_dir+inpfilt[ii]+'/'+str(tract1)+'/'+str(patch1)+'/calexp-'+inpfilt[ii]+'-'+str(tract1)+'-'+str(patch1)

                if os.path.isfile(input_image+'.fits') and not os.path.isfile(outdir1+'/'+rrh+ddh+'_'+filt[ii]+'_sci.fits'):
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

                    catalog[jj][inpfilt[ii]] = '%s-0-0'%dr

                    coadd = butler.get("deepCoadd_calexp", tract=tract.getId(),
                                       patch="%d,%d" % patch[0].getIndex(),
                                       filter=inpfilt[ii])  # your filter here

                    x=pixel[0]
                    y=pixel[1]

                    out_psffits = rrh+ddh+'_'+filt[ii]+'_psf.fits'

                    flagp=extract_posflag([input_image])
                    genpsfimage([input_image],flagp,[out_psffits])

            # removes huge fits file
            os.system('rm %s'%outfits)
                else:
                    print ra[jj],dec[jj], input_image,"not found"

outlines = []
for object in catalog:
    outline = '%s %f %f'%(object['name'], object['ra'], object['dec'])
    for filt in inpfilt:
        outline += ' %s'%object['filt']
    outline += '%s\n'%object['extra']
    outlines.append(outline)

f = open('summary.txt', 'w')
f.writelines(outlines)
f.close()

