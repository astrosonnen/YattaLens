from yattaconfig import *
import numpy as np
from pylens import pylens
from pylens import SBModels, MassModels
import emcee
from scipy.optimize import basinhopping
from scipy.stats import truncnorm


def fit_light(candidate, light_model, lfitband=('i'), nsamp=200, mask=None, rmax=20.):

    light_model.x.value = candidate.x
    light_model.y.value = candidate.y

    pars = [light_model.x, light_model.y, light_model.pa, light_model.q, light_model.re, light_model.n]
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

        for band in lfitband:
            logp, mags = pylens.getModel_lightonly(light_model.model[band], candidate.sci[band], candidate.err[band], candidate.X, \
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

    candidate.x = light_model.x.value
    candidate.y = light_model.y.value
    candidate.light_pa = light_model.pa.value
    candidate.light_q = light_model.q.value
    candidate.light_re = light_model.re.value
    candidate.light_n = light_model.n.value

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        logp, lmag, mimg = pylens.getModel_lightonly(light_model.model[band], candidate.sci[band], candidate.err[band], \
                                                     candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)
        resid = candidate.sci[band] - mimg

        candidate.lenssub_model[band] = mimg
        candidate.lenssub_resid[band] = resid


def quick_lens_subtraction(candidate, light_model, lfitband=('i'), niter=200, rmax=20.):

    pars = [light_model.x, light_model.y, light_model.pa, light_model.q, light_model.re]
    npars = len(pars)

    mask = (candidate.R < rmax).ravel()

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

        for band in lfitband:
            logp, mag = pylens.getModel_lightonly(light_model.model[band], candidate.sci[band], candidate.err[band], candidate.X, \
                                                  candidate.Y, zp=candidate.zp[band], mask=mask)

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

    candidate.x = light_model.x.value
    candidate.y = light_model.y.value
    candidate.light_pa = light_model.pa.value
    candidate.light_q = light_model.q.value
    candidate.light_re = light_model.re.value

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        logp, lmag, mimg = pylens.getModel_lightonly(light_model.model[band], candidate.sci[band], candidate.err[band], \
                                                     candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask)
        resid = candidate.sci[band] - mimg

        candidate.lenssub_model[band] = mimg
        candidate.lenssub_resid[band] = resid


def fit_ring(candidate, ring_model, light_model, image_set=None, rmax=30., nsamp=200):

    ring_model.pa.value = candidate.light_pa
    ring_model.q.value = candidate.light_q

    mask = np.ones(candidate.imshape)

    if image_set is not None:
        ring_model.rr.value = image_set['mean_arc_dist']
        for junk in image_set['junk']:
            mask[junk['footprint'] > 0] = 0

    mask[candidate.R > rmax] = 0

    mask_r = (mask > 0).ravel()

    pars = [ring_model.pa, ring_model.q, ring_model.rr, ring_model.hi, ring_model.ho]
    npars = len(pars)

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
            logp, mags = pylens.getModel_lightonly_ncomponents([light_model.model[band], ring_model.model[band]], \
                                                               candidate.sci[band], candidate.err[band], candidate.X, \
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

    print "fitting ring model..."

    sampler.run_mcmc(start, nsamp)

    candidate.ring_pars_sample = sampler.chain

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    candidate.ring_pa = ring_model.pa.value
    candidate.ring_q = ring_model.q.value
    candidate.ring_rr = ring_model.rr.value
    candidate.ring_hi = ring_model.hi.value
    candidate.ring_ho = ring_model.ho.value

    # removes the best-fit i-band model from all bands and saves the residuals
    chi2 = 0.
    for band in candidate.bands:

        logp, lmag, mimg = pylens.getModel_lightonly_ncomponents([light_model.model[band], ring_model.model[band]], candidate.sci[band], candidate.err[band], \
                                                     candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)

        candidate.ringfit_model[band] = mimg

        if band in fitband:
            chi2 += (((candidate.sci[band] - mimg[0] - mimg[1])/candidate.err[band])**2)[mask > 0].sum()

    candidate.ringfit_chi2 = chi2

def fit_lens(candidate, lens_model, light_model, image_set, rmax=30., nsamp=200):

    mask = np.ones(candidate.imshape)

    for junk in image_set['junk']:
        mask[junk['footprint'] > 0] = 0

    mask[candidate.R > rmax] = 0

    mask_r = (mask > 0).ravel()

    # guesses Einstein radius and source position
    narcs = len(image_set['arcs'])
    x_arcs = np.zeros(narcs)
    y_arcs = np.zeros(narcs)

    for i in range(narcs):
        x_arcs[i] = image_set['arcs'][i]['x']
        y_arcs[i] = image_set['arcs'][i]['y']

    if narcs > 1:
        lens_model.rein.value = image_set['mean_arc_dist']
    else:
        lens_model.rein.value = 0.7*image_set['mean_arc_dist']

    if candidate.light_q < 1.:
        lens_model.q.value = candidate.light_q
        lens_model.pa.value = candidate.light_pa
    else:
        lens_model.q.value = 1./candidate.light_q
        lens_model.pa.value = 1./candidate.light_pa

    lens_model.rein.parents['upper'] = image_set['furthest_arc']

    lens_model.lens.setPars()

    sx_guess, sy_guess = pylens.getDeflections(lens_model.lens, [x_arcs, y_arcs])

    lens_model.source_x.value = sx_guess.mean()
    lens_model.source_y.value = sy_guess.mean()

    pars = [lens_model.rein, lens_model.q, lens_model.pa, lens_model.source_x, lens_model.source_y]

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

    npars = len(pars)

    def logprior(allpars):
        for i in range(0, npars):
            if allpars[i] < bounds[i][0] or allpars[i] > bounds[i][1]:
                return -np.inf
        return 0.

    nwalkers = len(start)

    def logpfunc(allpars):
        lp = logprior(allpars)
        if not np.isfinite(lp):
            return -np.inf

        for j in range(0, npars):
            pars[j].value = allpars[j]
        sumlogp = 0.
        i = 0

        for band in fitband:
            logp, mags = pylens.getModel(lens_model.lens, light_model.model[band], lens_model.source[band], \
                                  candidate.sci[band], candidate.err[band], candidate.X, candidate.Y, \
                                  zp=candidate.zp[band], mask=mask_r)

            if logp != logp:
                return -np.inf
            sumlogp += logp
            i += 1

        return sumlogp

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

    print "fitting lens model..."

    sampler.run_mcmc(start, nsamp)

    output = {'chain': sampler.chain, 'logp': sampler.lnprobability}

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    candidate.lens_pa = lens_model.pa.value
    candidate.lens_q = lens_model.q.value
    candidate.lens_rein = lens_model.rein.value
    candidate.source_x = lens_model.source_x.value
    candidate.source_y = lens_model.source_y.value

    chi2 = 0.
    for band in candidate.bands:

        logp, lmag, mimg = pylens.getModel(lens_model.lens, light_model.model[band], lens_model.source[band], \
                                           candidate.sci[band], candidate.err[band], candidate.X, candidate.Y, zp=candidate.zp[band], \
                                           returnImg=True, mask=mask_r)

        candidate.lensfit_model[band] = mimg

        if band in fitband:
            chi2 += (((candidate.sci[band] - mimg[0] - mimg[1])/candidate.err[band])**2)[mask > 0].sum()

    candidate.lensfit_chi2 = chi2


