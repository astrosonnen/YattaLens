from yattaconfig import *
import numpy as np
import pyfits
import os
from photofit import indexTricks as iT, convolve
import pymc, emcee
from pylens import pylens, SBModels
from scipy.optimize import basinhopping
from scipy.stats import truncnorm


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


class Candidate:

    def __init__(self, name, bands=['i', 'r', 'g'], zp=[27., 27., 27.]):
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

        self.light_x = None
        self.light_y = None
        self.light_pa = None
        self.light_re = None
        self.light_q = None
        self.light_n = None

        self.light_model = {}
        self.lens_model = {}
        self.source_model = {}
        self.arc_models = []

        self.lenssub_model = {}
        self.lenssub_resid = {}

        self.objects = {}

    def read_data(self):

        print 'reading in data...'

        found = True
        shapeok = True

        s1 = -1
        s2 = -1

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
            self.x0 = float(s2/2)
            self.y0 = float(s1/2)

            Y, X = iT.coords(self.imshape)

            self.Y = Y
            self.X = X

            R = ((X - self.x0)**2 + (Y - self.y0)**2)**0.5
            self.R = R

            self.dataok = True

        else:
            self.dataok = False

        return self.dataok

    def prepare_model(self):

        print 'preparing model parameters...'

        self.lens_x = pymc.Uniform('x', lower=self.x0 - 2., upper=self.x0 + 2., value=self.x0)
        self.lens_y = pymc.Uniform('y', lower=self.y0 - 2., upper=self.y0 + 2., value=self.y0)
        self.light_pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.light_q = pymc.Uniform('q', lower=0.3, upper=2., value=0.7)
        self.light_re = pymc.Uniform('re', lower=3., upper=30., value=10.)
        self.light_n = pymc.Uniform('ns', lower=0.5, upper=8., value=4.)

        for band in self.bands:
            light = SBModels.Sersic('LensLight', {'x': self.lens_x, \
                                                'y': self.lens_y, \
                                                're': self.light_re, \
                                                'q': self.light_q, \
                                                'pa': self.light_pa, \
                                                'n': self.light_n})

            light.convolve = convolve.convolve(self.sci[band], self.psf[band])[1]

            self.light_model[band] = light

    def quick_lens_subtraction(self, rmax=20., niter=200, fitband=['i']):

        pars = [self.lens_x, self.lens_y, self.light_pa, self.light_q, self.light_re]
        npars = len(pars)

        mask = (self.R < rmax).ravel()

        bounds = []
        guess = []

        for par in pars:
            bounds.append((par.parents['lower'], par.parents['upper']))
            guess.append(par.value)

        barr = np.array(bounds)
        guess = np.array(guess)
        scale_free_bounds = 0.*barr
        scale_free_bounds[:,1] = 1.

        scale_free_guess = (guess - barr[:,0])/(barr[:,1] - barr[:,0])

        minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=1.)

        def logpfunc(scaledp):

            p = scaledp*(barr[:,1] - barr[:,0]) + barr[:,0]
            for j in range(0, npars):
                pars[j].value = p[j]
            sumlogp = 0.

            for band in fitband:
                logp, mag = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], self.X, \
                                                      self.Y, zp=self.zp[band], mask=mask)

                if logp != logp:
                    return -np.inf

                sumlogp += logp

            return -sumlogp

        print 'finding optimal lens light model...'
        res = basinhopping(logpfunc, scale_free_guess, stepsize=0.1, niter=niter, minimizer_kwargs=minimizer_kwargs, \
                           interval=30, T=3.)

        MLpars = res.x*(barr[:,1] - barr[:,0]) + barr[:,0]

        # updates values of the light parameters
        for j in range(npars):
            pars[j].value = MLpars[j]
            print MLpars[j]

        # removes the best-fit i-band model from all bands and saves the residuals
        for band in self.bands:
            self.light_model[band].setPars()

            logp, lmag, mimg = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], \
                                                         self.X, self.Y, zp=self.zp[band], returnImg=True, mask=mask)
            resid = self.sci[band] - mimg

            self.lenssub_model[band] = mimg
            self.lenssub_resid[band] = resid
            lsubname = modeldir+'/%s_%s_lenssub.fits'%(self.name, band)
            lmodname = modeldir+'/%s_%s_lensmod.fits'%(self.name, band)

            pyfits.PrimaryHDU(self.lenssub_resid[band]).writeto(lsubname, \
                                                            clobber=True)
            pyfits.PrimaryHDU(self.lenssub_model[band]).writeto(lmodname, \
                                                            clobber=True)

    def lens_subtraction(self, rmax=20., nsamp=200, fitband=['i'], mask=None):

        pars = [self.lens_x, self.lens_y, self.light_pa, self.light_q, self.light_re, self.light_n]
        npars = len(pars)

        if mask is None:
            mask = np.ones(self.imshape)

        mask[self.R > rmax] = 0

        mask_r = (mask > 0).ravel()

        bounds = []
        guess = []

        for par in pars:
            bounds.append((par.parents['lower'], par.parents['upper']))

        npars = len(pars)
        nwalkers = 6*npars

        def logprior(allpars):
            for i in range(0, npars):
                if allpars[i] < bounds[i][0] or allpars[i] > bounds[i][1]:
                    return -np.inf
            return 0.

        def logpfunc(allpars):
            lp = logprior(allpars)
            if not np.isfinite(lp):
                return -np.inf

            for j in range(0, npars):
                pars[j].value = allpars[j]
            sumlogp = 0.
            i = 0

            for band in fitband:
                logp, mags = pyfits.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], self.X, \
                                                       self.Y, zp=self.zp[band], mask=mask_r)

                if logp != logp:
                    return -np.inf
                sumlogp += logp
                i += 1

            return sumlogp

        sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

        start = []
        for i in range(nwalkers):
            tmp = np.zeros(npars)
            urand = np.random.rand(npars)
            for j in range(0, npars):
                p0 = urand[j]*(bounds[j][1] - bounds[j][0]) + bounds[j][0]
                tmp[j] = p0

            start.append(tmp)

        print "Sampling light profile parameters"

        sampler.run_mcmc(start, nsamp)

        self.light_pars_sample = sampler.chain

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # removes the best-fit i-band model from all bands and saves the residuals
        for band in self.bands:
            self.light_model[band].setPars()

            logp, lmag, mimg = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], \
                                                         self.X, self.Y, zp=self.zp[band], returnImg=True, mask=mask)
            resid = self.sci[band] - mimg

            self.lenssub_model[band] = mimg
            self.lenssub_resid[band] = resid

    def find_objects(self, detect_band='g', detect_thresh=3., meas_bands=['i', 'g'], model_err=0.2):

        lsubname = modeldir+'/%s_%s_lenssub.fits'%(self.name, detect_band)
        varname = datadir+'/%s_%s_var.fits'%(self.name, detect_band)

        segname = modeldir+'/%s_%s_segmap.fits'%(self.name, detect_band)
        catname = modeldir+'/%s_%s_secat.cat'%(self.name, detect_band)

        pyfits.PrimaryHDU(self.lenssub_resid[detect_band]).writeto(lsubname, \
                                                            clobber=True)

        os.system('sex %s -c seconfig.sex -WEIGHT_IMAGE %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -DETECT_THRESH %f'%\
                  (lsubname, varname, catname, segname, detect_thresh))

        f = open(catname, 'r')
        cat = np.atleast_2d(np.loadtxt(f))
        f.close()

        cs = cat.shape
        if cs[1] == 0:
            nobj = 0

        else:
            nobj = cs[0]

        segmap = pyfits.open(segname)[1].data.copy()

        objects = {}
        yesarcs = False

        for i in range(nobj):
            ind = i+1
            obj = {}
            x = cat[i, 0] - 1
            y = cat[i, 1] - 1
            obj['x'] = x
            obj['y'] = y
            obj['r'] = ((x - self.x0) ** 2 + (y - self.y0) ** 2) ** 0.5
            theta = np.rad2deg(np.arctan(-(x - self.x0) / (y - self.y0)))
            obj['ang_diff'] = min(abs(theta - cat[i, 5]), abs(abs(theta - cat[i, 5]) - 180.))

            if obj['r'] > minarcdist and obj['r'] < maxarcdist and obj['npix'] < maxarcsize \
                and obj['npix'] > minarcsize and obj['ang_diff'] < maxarcdang:
                yesarcs = True
                obj['arclike'] = True
            else:
                obj['arclike'] = False

            for band in meas_bands:
                obj['%s_flux'%band] = self.lenssub_resid[segmap==ind].sum()
                modeling_err = model_err*self.lenssub_model[segmap==ind].sum()
                obj['%s_err'%band] = (self.var[segmap==ind].sum() + modeling_err**2)**0.5

            objects[ind] = obj

        return objects, segmap, yesarcs
