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

        self.light_model = {}
        self.lens_model = None
        self.source_model = {}
        self.arc_models = []
        self.ring_model = {}

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

    def prepare_light_model(self):

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

    def prepare_ring_model(self):

        self.ring_pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.ring_q = pymc.Uniform('q', lower=0.2, upper=2., value=0.6)
        self.ring_rr = pymc.Uniform('rr', lower=0., upper=30., value=10.)
        self.ring_hi = pymc.Uniform('hi', lower=1., upper=30., value=5.)
        self.ring_ho = pymc.Uniform('ho', lower=1., upper=30., value=5.)

        for band in self.bands:
            ring = SBModels.Ring('Ring', {'x': self.lens_x, \
                                          'y': self.lens_y, \
                                          'rr': self.ring_rr, \
                                          'q': self.ring_q, \
                                          'pa': self.ring_pa, \
                                          'hi': self.ring_hi, \
                                          'ho': self.ring_ho,})

            ring.convolve = convolve.convolve(self.sci[band], self.psf[band])[1]

            self.ring_model[band] = ring

    def prepare_lens_model(self):

        self.lens_pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.lens_q = pymc.Uniform('q', lower=0.2, upper=1., value=0.6)
        self.lens_rein = pymc.Uniform('rein', lower=3., upper=30., value=10.)
        self.source_x = pymc.Uniform('hi', lower=self.x0 - source_range, upper=self.x0 + source_range, value=self.x0)
        self.source_y = pymc.Uniform('hi', lower=self.y0 - source_range, upper=self.y0 + source_range, value=self.y0)
        self.source_pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.source_q = pymc.Uniform('q', lower=0.2, upper=2., value=1.)
        self.source_re = pymc.Uniform('q', lower=0.5, upper=5., value=2.)

        self.lens_model = MassModels.PowerLaw('lens', {'x': self.lens_x, 'y': self.lens_y, 'b': self.lens_rein, \
                                                       'pa': self.lens_pa, 'q': self.lens_q, 'eta': 1.})

        for band in self.bands:
            source = SBModels.Sersic('Source', {'x': self.source_x, 'y': self.source_y, 're': self.source_re, \
                                                'pa': self.source_pa, 'q': self.source_q, 'n': 1.})
            source.convolve = convolve.convolve(self.sci[band], self.psf[band])[1]

            self.source_model[band] = source

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
            print guess[j], bounds[j], MLpars[j]

        # removes the best-fit i-band model from all bands and saves the residuals
        for band in self.bands:
            self.light_model[band].setPars()

            logp, lmag, mimg = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], \
                                                         self.X, self.Y, zp=self.zp[band], returnImg=True, mask=mask)
            resid = self.sci[band] - mimg

            self.lenssub_model[band] = mimg
            self.lenssub_resid[band] = resid

    def lens_subtraction_pymc(self, rmax=20., nsamp=10000, fitband=['i'], mask=None):

        pars = [self.lens_x, self.lens_y, self.light_pa, self.light_q, self.light_re, self.light_n]
        cov = [0.1, 0.1, 10., 0.01, 1., 0.01]

        npars = len(pars)

        if mask is None:
            mask = np.ones(self.imshape)

        mask[self.R > rmax] = 0

        mask_r = (mask > 0).ravel()

        npars = len(pars)

        print "sampling light profile parameters..."

        @pymc.deterministic
        def lightonlylogp(lp=pars):
            sumlogp = 0.
            for band in fitband:
                logp, mag = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], self.X, \
                                                      self.Y, zp=self.zp[band], mask=mask_r)

                if logp != logp:
                    return -1e300
                sumlogp += logp
            return sumlogp

        @pymc.stochastic(observed=True, name='logp')
        def logp(value=0., lp=pars):
            return lightonlylogp

        print 'sampling light parameters only'

        M = pymc.MCMC(pars + [lightonlylogp])
        M.use_step_method(pymc.AdaptiveMetropolis, pars, cov=np.diag(cov))
        M.sample(nsamp, 0)

        trace = {}

        for par in pars:
            trace[str(par)] = M.trace(par)[:]

        trace['logp'] = M.trace('lightonlylogp')[:]

        ML = trace['logp'].argmax()
        for i in range(npars):
            mlval = trace[str(pars[i])][ML]
            pars[i].value = mlval

        self.light_pars_sample = trace

        # removes the best-fit i-band model from all bands and saves the residuals
        for band in self.bands:
            self.light_model[band].setPars()

            logp, lmag, mimg = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], \
                                                         self.X, self.Y, zp=self.zp[band], returnImg=True, mask=mask_r)
            resid = self.sci[band] - mimg

            self.lenssub_model[band] = mimg
            self.lenssub_resid[band] = resid

    def lens_subtraction_emcee(self, rmax=20., nsamp=200, fitband=['i'], mask=None):

        pars = [self.lens_x, self.lens_y, self.light_pa, self.light_q, self.light_re, self.light_n]
        npars = len(pars)

        if mask is None:
            mask = np.ones(self.imshape)

        mask[self.R > rmax] = 0

        mask_r = (mask > 0).ravel()

        bounds = []

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
                logp, mags = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], self.X, \
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

        print "sampling light profile parameters..."

        sampler.run_mcmc(start, nsamp)

        self.light_pars_sample = sampler.chain

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # removes the best-fit i-band model from all bands and saves the residuals
        for band in self.bands:
            self.light_model[band].setPars()

            logp, lmag, mimg = pylens.getModel_lightonly(self.light_model[band], self.sci[band], self.err[band], \
                                                         self.X, self.Y, zp=self.zp[band], returnImg=True, mask=mask_r)
            resid = self.sci[band] - mimg

            self.lenssub_model[band] = mimg
            self.lenssub_resid[band] = resid

    def fit_ring_emcee(self, rmax=30., nsamp=200, mask=None):

        pars = [self.lens_x, self.lens_y, self.ring_pa, self.ring_q, self.ring_rr, self.ring_hi, self.ring_ho]
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
                logp, mags = pylens.getModel_lightonly_ncomponents([self.light_model[band], self.ring_model[band]], self.sci[band], self.err[band], self.X, \
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

        print "sampling light profile parameters..."

        sampler.run_mcmc(start, nsamp)

        self.ring_pars_sample = sampler.chain

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # removes the best-fit i-band model from all bands and saves the residuals
	chi2 = 0.
        for band in self.bands:
            self.ring_model[band].setPars()

            logp, lmag, mimg = pylens.getModel_lightonly_ncomponents([self.light_model[band], self.ring_model[band]], self.sci[band], self.err[band], \
                                                         self.X, self.Y, zp=self.zp[band], returnImg=True, mask=mask_r)

            self.ringfit_model[band] = mimg

	    if band in fitband:
		chi2 += (((self.sci[band] - mimg[0] - mimg[1])/self.err[band])**2)[mask > 0].sum()
		
	self.ringfit_chi2 = chi2

    def fit_lens(self, image_set, nstep=500, rmax=30.):

        mask = np.ones(self.imshape, dtype=int)

        for junk in image_set['junk']:
            mask[junk['footprint'] > 0] = 0

        mask[self.R > rmax] = 0

        mask_r = (mask > 0).ravel()

        # guesses Einstein radius and source position
        narcs = len(image_set['arcs'])
        x_arcs = np.zeros(narcs)
        y_arcs = np.zeros(narcs)
        farthest_arc = 0.
        mean_arcdist = 0.
        for i in range(narcs):
            x_arcs[i] = image_set['arcs'][i]['x']
            y_arcs[i] = image_set['arcs'][i]['y']
            mean_arcdist += image_set['arcs'][i]['r']
            if image_set['arcs'][i]['r'] > farthest_arc:
                farthest_arc = image_set['arcs'][i]['r']

        mean_arcdist /= float(narcs)

        if narcs > 1:
            self.lens_rein.value = mean_arcdist
        else:
            self.lens_rein.value = 0.7*mean_arcdist

        if self.light_q.value < 1.:
            self.lens_q.value = self.light_q.value
            self.lens_pa.value = self.light_pa.value
        else:
            self.lens_q.value = 1./self.light_q.value
            self.lens_pa.value = 1./self.light_pa.value

        self.lens_rein.parents['upper'] = farthest_arc

        self.lens_model.setPars()

        sx_guess, sy_guess = pylens.getDeflections(self.lens_model, [x_arcs, y_arcs])

        self.source_x.value = sx_guess.mean()
        self.source_y.value = sy_guess.mean()

        pars = [self.lens_rein, self.lens_q, self.lens_pa, self.source_x, self.source_y]
        step = [1., 0.1, 20., 1., 1.]

        npars = len(pars)

        bounds = []
        for par in pars:
            bounds.append((par.parents['lower'], par.parents['upper']))

        nwalkers = 50

        start = []
        for j in range(npars):
            a, b = (bounds[j][0] - pars[j].value)/step[j], (bounds[j][1] - pars[j].value)/step[j]
            tmp = truncnorm.rvs(a, b, size=nwalkers)*step[j] + pars[j].value

            start.append(tmp)

        start = np.array(start).T

        fitdic = pylens.do_fit_emcee_inputwalkers(self, pars, fitband, start, mask_r, nsamp=200)

        self.lens_model.setPars()

	chi2 = 0.
        for band in self.bands:
            self.source_model[band].setPars()

            logp, lmag, mimg = pylens.getModel(self.lens_model, self.light_model[band], self.source_model[band], \
                                               self.sci[band], self.err[band], self.X, self.Y, zp=self.zp[band], \
                                               returnImg=True, mask=mask_r)

            self.lensfit_model[band] = mimg

	    if band in fitband:
		chi2 += (((self.sci[band] - mimg[0] - mimg[1])/self.err[band])**2)[mask > 0].sum()
		
	self.lensfit_chi2 = chi2


