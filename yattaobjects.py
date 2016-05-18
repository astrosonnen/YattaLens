from yattaconfig import *
import numpy as np
import pyfits
import os
from photofit import indexTricks as iT, convolve
import pymc, emcee
from pylens import pylens, SBModels
from scipy.optimize import basinhopping
from scipy.stats import truncnorm
from pylens import MassModels


"""
class Object:
    def __init__(self, ind=ind, x=None, y=None, r=r, npix=None, ab=None, flux=None, err=None, tang=None, arc=None):

        self.ind = ind
        self.x = x
        self.y = y
        self.r = r
        self.npix = npix
        self.ab = ab
        self.flux = flux
        self.err = err
        self.
"""

class light_model:

    def __init__(self, candidate):

        self.x = pymc.Uniform('x', lower=candidate.x0 - 2., upper=candidate.x0 + 2., value=candidate.x0)
        self.y = pymc.Uniform('y', lower=candidate.y0 - 2., upper=candidate.y0 + 2., value=candidate.y0)
        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.3, upper=2., value=0.7)
        self.re = pymc.Uniform('re', lower=3., upper=30., value=10.)
        self.n = pymc.Uniform('ns', lower=0.5, upper=8., value=4.)

        self.model = {}

        for band in candidate.bands:
            light = SBModels.Sersic('LensLight', {'x': self.x, \
                                                'y': self.y, \
                                                're': self.re, \
                                                'q': self.q, \
                                                'pa': self.pa, \
                                                'n': self.n})

            light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = light


class ring_model:

    def __init__(self, candidate):

        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.2, upper=2., value=0.6)
        self.rr = pymc.Uniform('rr', lower=0., upper=30., value=10.)
        self.hi = pymc.Uniform('hi', lower=1., upper=30., value=5.)
        self.ho = pymc.Uniform('ho', lower=1., upper=30., value=5.)

        self.model = {}

        for band in candidate.bands:
            ring = SBModels.Ring('Ring', {'x': candidate.x, \
                                          'y': candidate.y, \
                                          'rr': self.rr, \
                                          'q': self.q, \
                                          'pa': self.pa, \
                                          'hi': self.hi, \
                                          'ho': self.ho,})

            ring.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = ring


class lens_model:

    def __init__(self, candidate):

        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.2, upper=1., value=0.6)
        self.rein = pymc.Uniform('rein', lower=3., upper=30., value=10.)
        self.source_x = pymc.Uniform('hi', lower=candidate.x0 - source_range, upper=candidate.x0 + source_range, value=candidate.x0)
        self.source_y = pymc.Uniform('hi', lower=candidate.y0 - source_range, upper=candidate.y0 + source_range, value=candidate.y0)
        self.source_pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.source_q = pymc.Uniform('q', lower=0.2, upper=2., value=1.)
        self.source_re = pymc.Uniform('q', lower=0.5, upper=5., value=2.)

        self.lens = MassModels.PowerLaw('lens', {'x': candidate.x, 'y': candidate.y, 'b': self.rein, \
                                                       'pa': self.pa, 'q': self.q, 'eta': 1.})

        self.source = {}

        for band in candidate.bands:
            source = SBModels.Sersic('Source', {'x': self.source_x, 'y': self.source_y, 're': self.source_re, \
                                                'pa': self.source_pa, 'q': self.source_q, 'n': 1.})

            source.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.source[band] = source


class Candidate:

    def __init__(self, name, bands=('i', 'r', 'g'), zp=(27., 27., 27.)):
        self.name = name
        self.bands = bands

        self.zp = {}
        i=0
        for band in self.bands:
            self.zp[band] = zp[i]

        self.x0 = None
        self.y0 = None
        self.X = None
        self.Y = None
        self.R = None
        self.sci = {}
        self.var = {}
        self.err = {}
        self.psf = {}
        self.imshape = None
        self.dataok = None

        self.x = None
        self.y = None
        self.light_pa = None
        self.light_re = None
        self.light_q = None
        self.light_n = None

        self.lens_pa = None
        self.lens_q = None
        self.lens_rein = None

        self.source_x = None
        self.source_y = None
        self.source_re = None
        self.source_pa = None
        self.source_q = None

        self.ring_pa = None
        self.ring_q = None
        self.ring_rr = None
        self.ring_hi = None
        self.ring_ho = None

        self.light_pars_sample = None
        self.ring_pars_sample = None
        self.lens_pars_sample = None

        self.lenssub_model = {}
        self.lenssub_resid = {}
        self.ringfit_model = {}
        self.lensfit_model = {}

        self.ringfit_chi2 = None
        self.lensfit_chi2 = None

        self.image_sets = None

        self.source_footprint = None

        self.ringfit_footprint_chi2 = None
        self.lensfit_footprint_chi2 = None

    def read_data(self):

        print 'reading in data...'

        found = True
        shapeok = True

        for band in self.bands:
            sciname = datadir+'/%s_%s_sci.fits' % (self.name, band)
            psfname = datadir+'/%s_%s_psf.fits' % (self.name, band)
            varname = datadir+'/%s_%s_var.fits' % (self.name, band)

            if os.path.isfile(sciname) and os.path.isfile(psfname) and os.path.isfile(varname):
                sci = pyfits.open(sciname)[0].data.copy()
                var = pyfits.open(varname)[0].data.copy()
                psf = pyfits.open(psfname)[1].data.copy()

                self.sci[band] = sci
                self.var[band] = var
                self.err[band] = var**0.5
                self.psf[band] = psf

                s = sci.shape

                if self.imshape is None:
                    self.imshape = s
                else:
                    if s != self.imshape:
                        shapeok = False

            else:
                found = False

        if found and shapeok:
            self.x0 = float(self.imshape[1]/2)
            self.y0 = float(self.imshape[0]/2)

            Y, X = iT.coords(self.imshape)

            self.Y = Y
            self.X = X

            R = ((X - self.x0)**2 + (Y - self.y0)**2)**0.5
            self.R = R

            self.dataok = True

        else:
            self.dataok = False

        return self.dataok

    def get_source_footprint(self, nsig=3.):

        footprint = np.zeros(self.imshape, dtype=int)

        for band in fitband:
            sdet = self.lensfit_model[band][1] > nsig*self.err[band]

            footprint[sdet] = 1

        self.source_footprint = footprint

    def get_footprint_chi2(self, image_set):

        lchi2 = 0.
        rchi2 = 0.

        mask = self.source_footprint

        for junk in image_set['junk']:
            mask[junk['footprint'] > 0] = 0

        mask[self.R > maxarcdist] = 0

        for band in fitband:
            lchi2 += (((self.sci[band] - self.lensfit_model[band][0] - self.lensfit_model[band][1])/self.err[band])**2)[mask > 0].sum()
            rchi2 += (((self.sci[band] - self.ringfit_model[band][0] - self.ringfit_model[band][1])/self.err[band])**2)[mask > 0].sum()

        self.lensfit_footprint_chi2 = lchi2
        self.ringfit_footprint_chi2 = rchi2

