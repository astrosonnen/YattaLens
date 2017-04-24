import numpy as np
import sys
from master_server_tools import labeling
import pickle
import pyfits


# this script loads CMASS and LOWZ catalogs from SDSS, matches it with the HSC catalog and writes a table with a bunch of info

racol = 0
deccol = 1
zcol = 7
idcol = 8
tractcol = 14
patchcol = 15
#fulldepthcol = 17

f = open('pdr1_sdss_dr12_catalog.csv', 'r')
radecz = np.loadtxt(f, usecols=(racol, deccol, zcol), delimiter=',')
f.close()

ra = radecz[:, 0]
dec = radecz[:, 1]
z = radecz[:, 2]

nobj = len(ra)

f = open('pdr1_sdss_dr12_catalog.csv', 'r')
idcol = np.loadtxt(f, usecols=(idcol, ), dtype=int, delimiter=',')
f.close()

f = open('pdr1_sdss_dr12_catalog.csv', 'r')
tractpatch = np.loadtxt(f, usecols=(tractcol, patchcol), dtype=int, delimiter=',')
f.close()

tract = tractpatch[:, 0]
patch = tractpatch[:, 1]

#f = open('pdr1_sdss_dr12_catalog.csv', 'r')
#fulldepth = np.loadtxt(f, usecols=(fulldepthcol, ), dtype=str, delimiter=',')
#f.close()

f = open('pdr1_sdss_dr12_catalog.csv', 'r')
grizy_inputs = np.genfromtxt(f, usecols=(16, 17, 18, 19, 20), dtype=int, delimiter=',', filling_values=0)
f.close()

fivebands = grizy_inputs.prod(axis=1)
fivebands[fivebands > 0] = 1

# reads in HSC magnitudes
f = open('pdr1_sdss_dr12_catalog.csv', 'r')
grizy = np.genfromtxt(f, usecols=(9, 10, 11, 12, 13), dtype=float, delimiter=',', filling_values=np.nan)
f.close()

outlines = []
outlines.append('# HSC_ID RA dec z_hsc z_sdss name sample plate mjd fiber gmag_aperture10 rmag_aperture10 imag_aperture10 zmag_aperture10 ymag_aperture10 tract patch fivebands\n')

for samp in ['cmass', 'lowz']:

    # reads in SDSS table
    sdss_table = pyfits.open('sdss_%s_sample_wpmf.fits'%samp)[1].data.copy()
    
    sdss_ra = sdss_table.field('RA').copy()
    sdss_dec = sdss_table.field('DEC').copy()
    sdss_plate = sdss_table.field('plate').copy()
    sdss_mjd = sdss_table.field('mjd').copy()
    sdss_fiberid = sdss_table.field('fiberid').copy()
    
    count = 0
    
    mindist = 2.
    
    coords = {}
    
    names = []
    ras = []
    decs = []
    zs = []
    plates = []
    mjds = []
    fiberids = []

    # goes through objects in HSC spec-z catalog and tries to match them to SDSS objects
    for i in range(nobj):
        dists = ((sdss_ra - ra[i])**2*np.cos(np.deg2rad(dec[i]))**2 + (sdss_dec - dec[i])**2)**0.5 * 3600.
        closest = dists.argmin()
        if dists[closest] < mindist:
    	    name = labeling.coords2name(ra[i], dec[i])
            outlines.append('%d %9.7f %9.7f %4.3f %4.3f %s %s %d %d %d %3.2f %3.2f %3.2f %3.2f %3.2f %d %d %d\n'%(idcol[i], ra[i], dec[i], z[i], sdss_table.field('z')[closest], name, samp.upper(), sdss_plate[closest], sdss_mjd[closest], sdss_fiberid[closest], grizy[i, 0], grizy[i, 1], grizy[i, 2], grizy[i, 3], grizy[i, 4], tract[i], patch[i], fivebands[i]))
    	    count += 1
   
f = open('hsc_pdr1_cmass_lowz.cat', 'w')
f.writelines(outlines)
f.close()

