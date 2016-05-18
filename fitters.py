from yattaconfig import *
import numpy as np
from pylens import pylens
from pylens import SBModels, MassModels
import emcee


def fit_light_profile(candidate, mask=None, lfitband=['i'], rmax=20., nsamp=200):

    x = pymc.Uniform('x', lower=candidate.x - 2., upper=candidate.x + 2., value=candidate.x)
    y = pymc.Uniform('y', lower=candidate.y - 2., upper=candidate.y + 2., value=candidate.y)
    light_pa = pymc.Uniform('pa', lower=-100., upper=100., value=candidate.light_pa)
    light_q = pymc.Uniform('q', lower=0.3, upper=2., value=candidate.light_q)
    light_re = pymc.Uniform('re', lower=3., upper=30., value=candidate.light_re)
    light_n = pymc.Uniform('n', lower=0.5, upper=8., value=candidate.light_n)

    light_model = {}
    for band in candidate.bands:
        light = SBModels.Sersic('LensLight', {'x': x, 'y': y, 'pa': light_pa, 'q': light_q, 're': light_re, 'n': light_n})

        light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

        light_model[band] = light

    pars = [x, y, light_pa, light_q, light_re, light_n]
    npars = len(pars)

    if mask is None:
        mask = np.ones(candidate.imshape)

    mask[candidate.R > rmax] = 0

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
            logp, mags = pylens.getModel_lightonly(light_model[band], candidate.sci[band], candidate.err[band], candidate.X, \
                                                   candidate.Y, zp=candidate.zp[band], mask=mask_r)

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

    candidate.light_pars_sample = sampler.chain

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:
        light_model[band].setPars()

        logp, lmag, mimg = pylens.getModel_lightonly(light_model[band], candidate.sci[band], candidate.err[band], \
                                                     candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)
        resid = candidate.sci[band] - mimg

        candidate.lenssub_model[band] = mimg
        candidate.lenssub_resid[band] = resid

