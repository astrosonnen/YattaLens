from yattaconfig import *
import numpy as np
from pylens import pylens
from pylens import SBModels, MassModels
import emcee
from scipy.optimize import basinhopping, nnls
from scipy.stats import truncnorm
from photofit import convolve


def fit_light(candidate, light_model, lfitband=('i'), guess=None, step=None, nsamp=200, mask=None, rmax=20.):

    pars = [light_model.x, light_model.y, light_model.pa, light_model.q, light_model.re, light_model.n]
    npars = len(pars)

    if guess is not None:
        print 'model starting point'
        if len(guess) == npars:
            for i in range(npars):
                pars[i].value = guess[i]
                if pars[i].parents['lower'] > guess[i]:
                    pars[i].parents['lower'] = guess[i]
                if pars[i].parents['upper'] < guess[i]:
                    pars[i].parents['upper'] = guess[i]
                print guess[i], (pars[i].parents['lower'], pars[i].parents['upper'])

    pars[2].parents['lower'] = pars[2].value - 100.
    pars[2].parents['upper'] = pars[2].value + 100.

    if step is None:
        step = [0.5, 0.5, 20., 0.1, 3., 0.3]

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
            logp, mags = pylens.getModel([], light_model.model[band], [], candidate.sci[band], candidate.err[band], \
                                         candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

            if logp != logp:
                return -np.inf
            sumlogp += logp
            i += 1

        return sumlogp

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

    start = []
    for j in range(npars):
        a, b = (bounds[j][0] - pars[j].value)/step[j], (bounds[j][1] - pars[j].value)/step[j]
        tmp = truncnorm.rvs(a, b, size=nwalkers)*step[j] + pars[j].value
        start.append(tmp)

    print "sampling light profile parameters..."

    sampler.run_mcmc(np.array(start).T, nsamp)

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

        resid = candidate.sci[band].copy()
        logp, lmag, mimgs = pylens.getModel([], light_model.model[band], [], candidate.sci[band], candidate.err[band], \
                                           candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)
        for mimg in mimgs:
            resid -= mimg

        candidate.lenssub_model[band] = mimgs
        candidate.lenssub_resid[band] = resid


def quick_lens_subtraction(candidate, light_model, lfitband=('i'), mask=None, guess=None, niter=200, rmax=20.):

    pars = [light_model.x, light_model.y, light_model.pa, light_model.q, light_model.re]
    npars = len(pars)

    if mask is None:
        mask = np.ones(candidate.imshape, dtype=int)

    mask[candidate.R > rmax] = 0
    mask_r = (mask > 0).ravel()

    bounds = []

    if guess is not None:
        print 'model starting point'
        if len(guess) == npars:
            for i in range(npars):
                pars[i].value = guess[i]
                if pars[i].parents['lower'] > guess[i]:
                    pars[i].parents['lower'] = guess[i]
                if pars[i].parents['upper'] < guess[i]:
                    pars[i].parents['upper'] = guess[i]
                print guess[i], (pars[i].parents['lower'], pars[i].parents['upper'])

    pars[2].parents['lower'] = pars[2].value - 100.
    pars[2].parents['upper'] = pars[2].value + 100.

    start = []
    for par in pars:
        bounds.append((par.parents['lower'], par.parents['upper']))
        start.append(par.value)

    barr = np.array(bounds)
    start = np.array(start)
    scale_free_bounds = 0.*barr
    scale_free_bounds[:,1] = 1.

    scale_free_guess = (start - barr[:,0])/(barr[:,1] - barr[:,0])

    minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=1.)

    def logpfunc(scaledp):

        p = scaledp*(barr[:,1] - barr[:,0]) + barr[:,0]
        for j in range(0, npars):
            pars[j].value = p[j]
        sumlogp = 0.

        for band in lfitband:
            logp, mag = pylens.getModel([], light_model.model[band], [], candidate.sci[band], candidate.err[band], candidate.X, \
                                                  candidate.Y, zp=candidate.zp[band], mask=mask_r)

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
        print start[j], bounds[j], MLpars[j]

    candidate.x = light_model.x.value
    candidate.y = light_model.y.value
    candidate.light_pa = light_model.pa.value
    candidate.light_q = light_model.q.value
    candidate.light_re = light_model.re.value

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        logp, lmag, mimg = pylens.getModel([], light_model.model[band], [], candidate.sci[band], candidate.err[band], \
                                                     candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)
        resid = candidate.sci[band] - mimg[0]

        candidate.lenssub_model[band] = mimg
        candidate.lenssub_resid[band] = resid

def fit_ring(candidate, ring_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

    ring_model.rr.parents['lower'] = 0.7*image_set['mean_arc_dist']
    ring_model.rr.parents['upper'] = min(30., 1.3*image_set['furthest_arc'])
    ring_model.rr.value = image_set['mean_arc_dist']

    ring_model.pa.value = candidate.light_pa
    ring_model.q.value = candidate.light_q

    allforegrounds = {}
    for band in candidate.bands:
        allforegrounds[band] = [light_model.model[band]]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                allforegrounds[band].append(comp['model'][band])
        for arc in foreground_model.bad_arcs:
            allforegrounds[band].append(arc['model'][band])

        allforegrounds[band].append(ring_model.model[band])

    mask = np.ones(candidate.imshape)
    mask[candidate.R > rmax] = 0

    for junk in image_set['junk']:
        mask[junk['footprint'] > 0] = 0

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
            logp, mags = pylens.getModel([], allforegrounds[band], [], candidate.sci[band], candidate.err[band], \
                                         candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

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

        logp, lmag, mimgs = pylens.getModel([], allforegrounds[band], [], candidate.sci[band], candidate.err[band], \
                                            candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, \
                                            mask=mask_r)

        candidate.ringfit_model[band] = mimgs

        if band in fitband:
            resid = candidate.sci[band].copy()
            for mimg in mimgs:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.ringfit_chi2 = chi2
    candidate.ringfit_mask = mask


def fit_new_ring(candidate, ring_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

    ring_model.rr.parents['lower'] = 0.7*image_set['mean_arc_dist']
    ring_model.rr.parents['upper'] = min(30., 1.3*image_set['furthest_arc'])
    ring_model.rr.value = image_set['mean_arc_dist']

    ring_model.pa.value = candidate.light_pa
    ring_model.q.value = candidate.light_q

    allforegrounds = {}
    for band in candidate.bands:
        allforegrounds[band] = [light_model.model[band]]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                allforegrounds[band].append(comp['model'][band])
        for arc in foreground_model.bad_arcs:
            allforegrounds[band].append(arc['model'][band])

        allforegrounds[band].append(ring_model.model[band])

    mask = np.ones(candidate.imshape)
    mask[candidate.R > rmax] = 0

    for junk in image_set['junk']:
        mask[junk['footprint'] > 0] = 0

    mask_r = (mask > 0).ravel()

    pars = [ring_model.pa, ring_model.q, ring_model.rr, ring_model.width, ring_model.smooth]
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
            logp, mags = pylens.getModel([], allforegrounds[band], [], candidate.sci[band], candidate.err[band], \
                                         candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

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
    candidate.ring_smooth = ring_model.smooth.value
    candidate.ring_width = ring_model.width.value

    # removes the best-fit i-band model from all bands and saves the residuals
    chi2 = 0.
    for band in candidate.bands:

        logp, lmag, mimgs = pylens.getModel([], allforegrounds[band], [], candidate.sci[band], candidate.err[band], \
                                            candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, \
                                            mask=mask_r)

        candidate.ringfit_model[band] = mimgs

        if band in fitband:
            resid = candidate.sci[band].copy()
            for mimg in mimgs:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.ringfit_chi2 = chi2
    candidate.ringfit_mask = mask


def fit_sersic(candidate, sersic_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

    x_brightest = -1.
    y_brightest = -1.
    size_brightest = 0
    brightest = -np.inf

    for arc in image_set['arcs']:
        if arc['g_flux'] > brightest:
            brightest = arc['g_flux']
            x_brightest = arc['x']
            y_brightest = arc['y']
            size_brightest = arc['npix']

    for image in image_set['images']:
        if image['g_flux'] > brightest:
            brightest = image['g_flux']
            x_brightest = image['x']
            y_brightest = image['y']
            size_brightest = image['npix']

    re_brightest = (size_brightest/np.pi)**0.5

    theta = np.rad2deg(np.arctan((y_brightest - candidate.y)/(x_brightest - candidate.x)))

    sersic_model.x.value = x_brightest
    sersic_model.y.value = y_brightest
    sersic_model.x.parents['lower'] = x_brightest - 3.
    sersic_model.x.parents['upper'] = x_brightest + 3.

    sersic_model.y.parents['lower'] = y_brightest - 3.
    sersic_model.y.parents['upper'] = y_brightest + 3.

    sersic_model.pa.value = theta + 90.
    sersic_model.pa.parents['lower'] = theta + 60.
    sersic_model.pa.parents['upper'] = theta + 120.

    sersic_model.re.parents['upper'] = re_brightest
    sersic_model.re.value = re_brightest

    allforegrounds = {}
    for band in candidate.bands:
        allforegrounds[band] = [light_model.model[band]]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                allforegrounds[band].append(comp['model'][band])
        for arc in foreground_model.bad_arcs:
            allforegrounds[band].append(arc['model'][band])

        allforegrounds[band].append(sersic_model.model[band])

    mask = np.ones(candidate.imshape)
    mask[candidate.R > rmax] = 0

    for junk in image_set['junk']:
        mask[junk['footprint'] > 0] = 0

    mask_r = (mask > 0).ravel()

    pars = [sersic_model.x, sersic_model.y, sersic_model.pa, sersic_model.q, sersic_model.re, sersic_model.b4]

    npars = len(pars)

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
            logp, mags = pylens.getModel([], allforegrounds[band], [], candidate.sci[band], candidate.err[band], \
                                         candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

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

    print "fitting sersic model..."

    sampler.run_mcmc(start, nsamp)

    candidate.sersic_pars_sample = sampler.chain

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    candidate.sersic_pa = sersic_model.pa.value
    candidate.sersic_q = sersic_model.q.value
    candidate.sersic_x = sersic_model.x.value
    candidate.sersic_y = sersic_model.y.value
    candidate.sersic_re = sersic_model.re.value
    candidate.sersic_b4 = sersic_model.b4.value

    # removes the best-fit i-band model from all bands and saves the residuals
    chi2 = 0.
    for band in candidate.bands:

        logp, lmag, mimgs = pylens.getModel([], allforegrounds[band], [], candidate.sci[band], candidate.err[band], \
                                            candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, \
                                            mask=mask_r)

        candidate.sersicfit_model[band] = mimgs

        if band in fitband:
            resid = candidate.sci[band].copy()
            for mimg in mimgs:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.sersicfit_chi2 = chi2
    candidate.sersicfit_mask = mask


def fit_foregrounds(candidate, foreground_model, light_model, lfitband=(lightband), rmax=30., nsamp=100):

    allmodels = {}
    for band in candidate.bands:
        allmodels[band] = [light_model.model[band]]

    mask = np.ones(candidate.imshape, dtype=int)
    mask[candidate.R > rmax] = 0
    mask_r = (mask > 0).ravel()

    count = 1
    for comp in foreground_model.components:

        tmp_mask = comp['mask'].copy()
        tmp_mask[candidate.R > rmax] = 0
        mask_r = (tmp_mask > 0).ravel()

        for band in candidate.bands:
            allmodels[band].append(comp['model'][band])

        pars = comp['pars']

        npars = len(pars)

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
                logp, mags = pylens.getModel([], allmodels[band], [], candidate.sci[band], candidate.err[band], \
                                             candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

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

        print "fitting foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value)

        count += 1

        sampler.run_mcmc(start, nsamp)

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        logp, lmag, mimgs = pylens.getModel([], allmodels[band], [], candidate.sci[band], candidate.err[band], \
                                           candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)

        resid = candidate.sci[band].copy()
        for mimg in mimgs:
            resid -= mimg

        candidate.foreground_model[band] = mimgs


def fit_foregrounds_fixedamps(candidate, foreground_model, light_model, lfitband=(lightband), rmax=30., nsamp=100):

    allmodels = {}
    allamps = {}

    for band in candidate.bands:
        allmodels[band] = [light_model.model[band]]
        allamps[band] = [1.]
        foreground_model.amps[band] = []

    mask = np.ones(candidate.imshape, dtype=int)
    mask[candidate.R > rmax] = 0
    mask_r = (mask > 0).ravel()

    count = 1
    for comp in foreground_model.components:

        tmp_mask = comp['mask'].copy()
        tmp_mask[candidate.R > rmax] = 0
        mask_r = (tmp_mask > 0).ravel()

        for band in candidate.bands:
            allmodels[band].append(comp['model'][band])

        pars = comp['pars']

        comp['amps'] = {}

        npars = len(pars)

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

            for band in lfitband:

                lmodel = 0.*candidate.sci[band].ravel()
                modlist = []
                for i in range(count):
                    allmodels[band][i].amp = allamps[band][i]
                    allmodels[band][i].setPars()
                    lmodel += (convolve.convolve(allmodels[band].pixeval(candidate.X, candidate.Y), \
                                            allmodels[band].convolve, False)[0].ravel()/(candidate.sig[band].ravel()))

                modlist.append(lmodel)

                allmodels[band][count].amp = 1.
                allmodels[band][count].setPars()

                lmodel = (convolve.convolve(allmodels[band][count].pixeval(candidate.X, candidate.Y), \
                                            allmodels[band][count].convolve, \
                                            False)[0].ravel()/(candidate.sig[band].ravel()))

                modlist.append(lmodel)

                modarr = np.array(modlist).T

                amps, chi = nnls(modarr, (candidate.sci[band]/candidate.sig[band]).ravel()[mask_r])

                logp = -0.5*chi

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

        print "fitting foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value)


        sampler.run_mcmc(start, nsamp)

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # fixes the amplitude to the best fit value
        for band in lfitband:

            lmodel = 0.*candidate.sci[band].ravel()
            modlist = []
            for i in range(count):
                allmodels[band][i].amp = allamps[band][i]
                allmodels[band][i].setPars()
                lmodel += (convolve.convolve(allmodels[band].pixeval(candidate.X, candidate.Y), \
                                        allmodels[band].convolve, False)[0].ravel()/(candidate.sig[band].ravel()))

            modlist.append(lmodel)

            allmodels[band][count].amp = 1.
            allmodels[band][count].setPars()

            lmodel = (convolve.convolve(allmodels[band][count].pixeval(candidate.X, candidate.Y), \
                                        allmodels[band][count].convolve, \
                                        False)[0].ravel()/(candidate.sig[band].ravel()))

            modlist.append(lmodel)

            modarr = np.array(modlist).T

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.sig[band]).ravel()[mask_r])

            allamps[band].append(amps[1]/amps[0])

        count += 1

    # removes the best-fit i-band model from all bands, saves the residuals and fixes the amplitude of the foreground
    for band in candidate.bands:

        lmodel = 0.*candidate.sci[band]
        for i in range(count):
            allmodels[band][i].amp = allamps[band][i]
            allmodels[band][i].setPars()
            lmodel += (convolve.convolve(allmodels[band].pixeval(candidate.X, candidate.Y), \
                                    allmodels[band].convolve, False)[0])

        fitmodel = np.atleast_2d((lmodel/candidate.sig[band]).ravel()[mask_r]).T

        amps, chi = nnls(fitmodel, (candidate.sci[band]/candidate.sig[band]).ravel()[mask_r])

        logp = -0.5*chi

        resid = candidate.sci[band].copy()
        resid -= lmodel*amps[0]

        candidate.foreground_model[band] = lmodel*amps[0]


def fit_bad_arcs(candidate, foreground_model, light_model, rmax=30., nsamp=200):

    allmodels = {}
    for band in candidate.bands:
        allmodels[band] = [light_model.model[band]]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                allmodels[band].append(comp['model'][band])

    count = 1
    for arc in foreground_model.bad_arcs:

        tmp_mask = np.ones(candidate.imshape, dtype=int)
        tmp_mask[candidate.R > rmax] = 0
        mask_r = (tmp_mask > 0).ravel()

        for band in candidate.bands:
            allmodels[band].append(arc['model'][band])

        pars = arc['pars']

        npars = len(pars)

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
                logp, mags = pylens.getModel([], allmodels[band], [], candidate.sci[band], candidate.err[band], \
                                             candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

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

        print "fitting arc-like foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value)

        count += 1
        sampler.run_mcmc(start, nsamp)

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        logp, lmag, mimgs = pylens.getModel([], allmodels[band], [], candidate.sci[band], candidate.err[band], \
                                           candidate.X, candidate.Y, zp=candidate.zp[band], returnImg=True, mask=mask_r)

        resid = candidate.sci[band].copy()
        for mimg in mimgs:
            resid -= mimg

        candidate.foreground_model[band] = mimgs


def fit_lens(candidate, lens_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

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
        if candidate.light_q < 0.7:
            lens_model.pa.parents['lower'] = lens_model.pa.value - 30.
            lens_model.pa.parents['upper'] = lens_model.pa.value + 30.
        else:
            lens_model.q.parents['lower'] = 0.5
    else:
        lens_model.q.value = 1./candidate.light_q
        lens_model.pa.value = candidate.light_pa + 90.
        if candidate.light_q > 1./0.7:
            lens_model.pa.parents['lower'] = lens_model.pa.value - 30.
            lens_model.pa.parents['upper'] = lens_model.pa.value + 30.
        else:
            lens_model.q.parents['lower'] = 0.5

    lens_model.rein.parents['upper'] = image_set['furthest_arc']

    lens_model.lens.setPars()

    sx_guess, sy_guess = pylens.getDeflections(lens_model.lens, [x_arcs, y_arcs])

    lens_model.source_x.value = sx_guess.mean()
    lens_model.source_y.value = sy_guess.mean()

    pars = [lens_model.rein, lens_model.q, lens_model.pa, lens_model.source_x, lens_model.source_y, \
            lens_model.source_re]

    step = [1., 0.1, 20., 1., 1., 0.1]

    npars = len(pars)

    bounds = []
    for par in pars:
        bounds.append((par.parents['lower'], par.parents['upper']))

    nwalkers = 50

    allforegrounds = {}
    allamps = {}
    for band in candidate.bands:
        allforegrounds[band] = [light_model.model[band]]
        allamps[band] = [1.]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                allforegrounds[band].append(comp['model'][band])
                allamps[band].append(foreground_model.amps[band][i])
        for arc in foreground_model.bad_arcs:
            allforegrounds[band].append(arc['model'][band])

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
            logp, mags = pylens.getModel_fixedamps([lens_model.lens], allforegrounds[band], [lens_model.source[band]], \
                                                   allamps[band], candidate.sci[band], candidate.err[band], \
                                                   candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)

            if logp != logp:
                return -np.inf
            sumlogp += logp
            i += 1

        return sumlogp

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

    print "fitting lens model..."

    sampler.run_mcmc(start, nsamp)

    candidate.lens_pars_sample = sampler.chain

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    candidate.lens_pa = lens_model.pa.value
    candidate.lens_q = lens_model.q.value
    candidate.lens_rein = lens_model.rein.value
    candidate.source_x = lens_model.source_x.value
    candidate.source_y = lens_model.source_y.value
    candidate.source_re = lens_model.source_re.value

    chi2 = 0.
    for band in candidate.bands:

        logp, lmag, mimgs = pylens.getModel(lens_model.lens, allforegrounds[band], lens_model.source[band], \
                                           candidate.sci[band], candidate.err[band], candidate.X, candidate.Y, \
                                           zp=candidate.zp[band], returnImg=True, mask=mask_r)

        candidate.lensfit_model[band] = mimgs

        if band in fitband:
            resid = candidate.sci[band].copy()
            for mimg in mimgs:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.lensfit_chi2 = chi2
    candidate.lensfit_mask = mask


