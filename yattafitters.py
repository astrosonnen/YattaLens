from yattaconfig import def_config
import numpy as np
from yattapylens import pylens
from yattapylens import SBModels, MassModels
import emcee
from scipy.optimize import basinhopping, nnls
from scipy.stats import truncnorm
from photofit import convolve


def fit_light(candidate, light_model, lfitband=('i'), guess=None, step=None, nsamp=200, mask=None, rmax=20., fit_method='minimize'):

    pars = [light_model.x, light_model.y, light_model.pa, light_model.q, light_model.re, light_model.n]
    npars = len(pars)

    if guess is not None:
        print('model starting point')
        if len(guess) == npars:
            for i in range(npars):
                pars[i].value = guess[i]
                if pars[i].lower > guess[i]:
                    pars[i].lower = guess[i]
                if pars[i].upper < guess[i]:
                    pars[i].upper = guess[i]
                print(guess[i], (pars[i].lower, pars[i].upper))

    pars[2].lower = pars[2].value - 100.
    pars[2].upper = pars[2].value + 100.

    if step is None:
        step = [0.5, 0.5, 20., 0.1, 3., 0.3]

    if mask is None:
        mask = np.ones(candidate.imshape)

    mask[candidate.R > rmax] = 0

    mask_r = (mask > 0).ravel()

    bounds = []

    for par in pars:
        bounds.append((par.lower, par.upper))

    if fit_method == 'MCMC':

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
    
        print("sampling light profile parameters...")
    
        sampler.run_mcmc(np.array(start).T, nsamp)
    
        candidate.light_pars_sample = sampler.chain
    
        ML = sampler.flatlnprobability.argmax()
    
        for j in range(npars):
            pars[j].value = sampler.flatchain[ML, j]

    elif fit_method == 'minimize':

        start = []
        for j in range(npars):
            start.append(pars[j].value)

        barr = np.array(bounds)
        start = np.array(start)
        scale_free_bounds = 0.*barr
        scale_free_bounds[:,1] = 1.
    
        scale_free_guess = (start - barr[:,0])/(barr[:,1] - barr[:,0])
    
        minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=1.)
    
        def nlogpfunc(scaledp):
    
            p = scaledp*(barr[:,1] - barr[:,0]) + barr[:,0]
            for j in range(0, npars):
                pars[j].value = p[j]
            sumlogp = 0.
    
            for band in lfitband:
                logp, mags = pylens.getModel([], light_model.model[band], [], candidate.sci[band], candidate.err[band], \
                                             candidate.X, candidate.Y, zp=candidate.zp[band], mask=mask_r)
     
                if logp != logp:
                    return -np.inf
    
                sumlogp += logp
    
            return -sumlogp
    
        print('finding optimal lens light model...')
        res = basinhopping(nlogpfunc, scale_free_guess, stepsize=0.1, niter=nsamp, minimizer_kwargs=minimizer_kwargs, \
                           interval=30, T=3.)

        MLpars = res.x*(barr[:,1] - barr[:,0]) + barr[:,0]
    
        # updates values of the light parameters
        for j in range(npars):
            pars[j].value = MLpars[j]

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


def fit_ring(candidate, ring_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

    ring_model.rr.lower = 0.7*image_set['mean_arc_dist']
    ring_model.rr.upper = min(30., 1.3*image_set['furthest_arc'])
    ring_model.rr.value = image_set['mean_arc_dist']

    ring_model.pa.value = candidate.light_pa
    ring_model.q.value = candidate.light_q

    foregrounds = {}
    for band in candidate.bands:
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]

        foregrounds[band]  = [lmodel]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                foregrounds[band].append(comp['scalefreemodel'][band])

        for arc in foreground_model.bad_arcs:
            foregrounds[band].append(arc['scalefreemodel'][band])

        for obj in foreground_model.new_foregrounds:
            foregrounds[band].append(obj['scalefreemodel'][band])

    mask = np.ones(candidate.imshape)
    mask[candidate.R > rmax] = 0

    for junk in image_set['junk']:
        mask[junk['footprint'] > 0] = 0

    mask_r = (mask > 0).ravel()

    pars = [ring_model.pa, ring_model.q, ring_model.rr, ring_model.hi, ring_model.ho]
    npars = len(pars)

    bounds = []

    for par in pars:
        bounds.append((par.lower, par.upper))

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

        for band in candidate.fitband:

            modlist = []
            fixedcomps = 0.*candidate.sci[band]
            for l in foregrounds[band]:
                fixedcomps += l

            modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

            ring_model.model[band].setPars()
            ring_model.model[band].amp = 1.
            rmodel = convolve.convolve(ring_model.model[band].pixeval(candidate.X, candidate.Y), \
                                       ring_model.model[band].convolve, False)[0]

            modlist.append((rmodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            if np.isnan(modarr).any():
                return -1e300

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

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

    print("fitting ring model...")

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

        modlist = []
        fixedcomps = 0.*candidate.sci[band]
        for l in foregrounds[band]:
            fixedcomps += l

        modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

        ring_model.model[band].setPars()
        ring_model.model[band].amp = 1.
        rmodel = convolve.convolve(ring_model.model[band].pixeval(candidate.X, candidate.Y), \
                                   ring_model.model[band].convolve, False)[0]

        modlist.append((rmodel/candidate.err[band]).ravel()[mask_r])

        modarr = np.array(modlist).T

        amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        candidate.ringfit_model[band] = []
        for l in foregrounds[band]:
            candidate.ringfit_model[band].append(amps[0]*l)

        candidate.ringfit_model[band].append(amps[1]*rmodel)

        if band in candidate.fitband:
            resid = candidate.sci[band].copy()
            for mimg in candidate.ringfit_model[band]:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.ringfit_chi2 = chi2
    candidate.ringfit_mask = mask


def fit_new_ring(candidate, ring_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

    ring_model.rr.lower = 0.7*image_set['mean_arc_dist']
    ring_model.rr.upper = min(30., 1.3*image_set['furthest_arc'])
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
        bounds.append((par.lower, par.upper))

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

        for band in candidate.fitband:
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

    print("fitting ring model...")

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

        if band in candidate.fitband:
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
        if arc['%s_flux'%candidate.fitband] > brightest:
            brightest = arc['%s_flux'%candidate.fitband]
            x_brightest = arc['x']
            y_brightest = arc['y']
            size_brightest = arc['npix']

    for image in image_set['images']:
        if image['%s_flux'%candidate.fitband] > brightest:
            brightest = image['%s_flux'%candidate.fitband]
            x_brightest = image['x']
            y_brightest = image['y']
            size_brightest = image['npix']

    re_brightest = (size_brightest/np.pi)**0.5

    theta = np.rad2deg(np.arctan((y_brightest - candidate.y)/(x_brightest - candidate.x)))

    sersic_model.x.value = x_brightest
    sersic_model.y.value = y_brightest
    sersic_model.x.lower = x_brightest - 3.
    sersic_model.x.upper = x_brightest + 3.

    sersic_model.y.lower = y_brightest - 3.
    sersic_model.y.upper = y_brightest + 3.

    sersic_model.pa.value = theta + 90.
    sersic_model.pa.lower = theta + 60.
    sersic_model.pa.upper = theta + 120.

    sersic_model.re.upper = re_brightest
    sersic_model.re.value = re_brightest

    foregrounds = {}
    for band in candidate.bands:
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]

        foregrounds[band]  = [lmodel]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                foregrounds[band].append(comp['scalefreemodel'][band])

        for arc in foreground_model.bad_arcs:
            foregrounds[band].append(arc['scalefreemodel'][band])

        for obj in foreground_model.new_foregrounds:
            foregrounds[band].append(obj['scalefreemodel'][band])

    mask = np.ones(candidate.imshape)
    mask[candidate.R > rmax] = 0

    for junk in image_set['junk']:
        mask[junk['footprint'] > 0] = 0

    mask_r = (mask > 0).ravel()

    pars = [sersic_model.x, sersic_model.y, sersic_model.pa, sersic_model.q, sersic_model.re, sersic_model.b4]

    npars = len(pars)

    bounds = []

    for par in pars:
        bounds.append((par.lower, par.upper))

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

        for band in candidate.fitband:

            modlist = []
            fixedcomps = 0.*candidate.sci[band]
            for l in foregrounds[band]:
                fixedcomps += l

            modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

            sersic_model.model[band].setPars()
            sersic_model.model[band].amp = 1.
            smodel = convolve.convolve(sersic_model.model[band].pixeval(candidate.X, candidate.Y), \
                                       sersic_model.model[band].convolve, False)[0]

            modlist.append((smodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            if np.isnan(modarr).any():
                return -1e300

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

            logp = -0.5*chi

            if logp != logp:
                return -np.inf
            sumlogp += logp

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

    print("fitting sersic model...")

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

        modlist = []
        fixedcomps = 0.*candidate.sci[band]
        for l in foregrounds[band]:
            fixedcomps += l

        modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

        sersic_model.model[band].setPars()
        sersic_model.model[band].amp = 1.
        smodel = convolve.convolve(sersic_model.model[band].pixeval(candidate.X, candidate.Y), \
                                   sersic_model.model[band].convolve, False)[0]

        modlist.append((smodel/candidate.err[band]).ravel()[mask_r])

        modarr = np.array(modlist).T

        amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        candidate.sersicfit_model[band] = []
        for l in foregrounds[band]:
            candidate.sersicfit_model[band].append(amps[0]*l)

        candidate.sersicfit_model[band].append(amps[1]*smodel)

        if band in candidate.fitband:
            resid = candidate.sci[band].copy()
            for mimg in candidate.sersicfit_model[band]:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.sersicfit_chi2 = chi2
    candidate.sersicfit_mask = mask


def fit_foregrounds(candidate, foreground_model, light_model, rmax=30., nsamp=100):

    lfitband = candidate.lightband

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
            bounds.append((par.lower, par.upper))

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

        print("fitting foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value))

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


def fit_foregrounds_fixedamps(candidate, foreground_model, light_model, rmax=30., nsamp=100):

    lfitband = candidate.lightband

    scalefreemodels = {}

    for band in candidate.bands:
        #foreground_model.amps[band] = []
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]
        scalefreemodels[band] = [lmodel]

    mask = np.ones(candidate.imshape, dtype=int)
    mask[candidate.R > rmax] = 0
    mask_r = (mask > 0).ravel()

    count = 1
    for comp in foreground_model.components:

        tmp_mask = comp['mask'].copy()
        tmp_mask[candidate.R > rmax] = 0
        mask_r = (tmp_mask > 0).ravel()

        pars = comp['pars']

        comp['scalefreemodel'] = {}
        comp['unitampmodel'] = {}
        comp['scalefreemags'] = {}

        npars = len(pars)

        bounds = []

        for par in pars:
            bounds.append((par.lower, par.upper))

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

                modlist = []
                fixedcomps = 0.*candidate.sci[band]
                for l in scalefreemodels[band]:
                    fixedcomps += l

                modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

                comp['model'][band].amp = 1.
                comp['model'][band].setPars()
                lmodel = convolve.convolve(comp['model'][band].pixeval(candidate.X, candidate.Y), \
                                            comp['model'][band].convolve, False)[0]

                modlist.append((lmodel/candidate.err[band]).ravel()[mask_r])

                modarr = np.array(modlist).T

                amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

                logp = -0.5*chi

                if logp != logp:
                    return -np.inf
                sumlogp += logp

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

        print("fitting foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value))


        sampler.run_mcmc(start, nsamp)

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # fixes the amplitude to the best fit value, in each band
        for band in candidate.bands:

            modlist = []
            fixedcomps = 0.*candidate.sci[band]
            for l in scalefreemodels[band]:
                fixedcomps += l

            modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

            comp['model'][band].amp = 1.
            comp['model'][band].setPars()
            lmodel = convolve.convolve(comp['model'][band].pixeval(candidate.X, candidate.Y), \
                                        comp['model'][band].convolve, False)[0]
            comp['unitampmodel'][band] = lmodel

            modlist.append((lmodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

            if amps[0] == 0.:
                amps[0] = 1.

            comp['scalefreemodel'][band] = amps[1]/amps[0]*lmodel

            if amps[1] <= 0.:
                mag = 99.
            else:
                mag = comp['model'][band].Mag(candidate.zp[band]) - 2.5*np.log10(amps[1]/amps[0])

            comp['scalefreemags'][band] = mag

            scalefreemodels[band].append(amps[1]/amps[0]*lmodel)

        count += 1

    # removes the best-fit i-band model from all bands, saves the residuals and fixes the amplitude of the foreground
    for band in candidate.bands:

        lmodel = 0.*candidate.sci[band]
        for i in range(count):
            lmodel += scalefreemodels[band][i]

        fitmodel = np.atleast_2d((lmodel/candidate.err[band]).ravel()[mask_r]).T

        amps, chi = nnls(fitmodel, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        logp = -0.5*chi

        resid = candidate.sci[band].copy()
        resid -= lmodel*amps[0]

        candidate.foreground_model[band] = []
        for i in range(count):
            candidate.foreground_model[band].append(amps[0]*scalefreemodels[band][i])

        #candidate.foreground_model[band] = lmodel*amps[0]


def fit_bad_arcs(candidate, foreground_model, light_model, rmax=30., nsamp=200):

    scalefreemodels = {}

    for band in candidate.bands:
        #foreground_model.amps[band] = []
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]
        scalefreemodels[band] = [lmodel]

        for comp in foreground_model.components:
            if comp['dofit'] == True:
                scalefreemodels[band].append(comp['scalefreemodel'][band])

    count = len(scalefreemodels[band])

    for arc in foreground_model.bad_arcs:

        tmp_mask = np.ones(candidate.imshape, dtype=int)
        tmp_mask[candidate.R > rmax] = 0
        mask_r = (tmp_mask > 0).ravel()

        pars = arc['pars']

        arc['scalefreemodel'] = {}
        arc['scalefreemags'] = {}
        arc['unitampmodel'] = {}

        npars = len(pars)

        bounds = []

        for par in pars:
            bounds.append((par.lower, par.upper))

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

            for band in candidate.fitband:

                modlist = []
                fixedcomps = 0.*candidate.sci[band]
                for l in scalefreemodels[band]:
                    fixedcomps += l

                modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

                arc['model'][band].amp = 1.
                arc['model'][band].setPars()
                lmodel = convolve.convolve(arc['model'][band].pixeval(candidate.X, candidate.Y), \
                                            arc['model'][band].convolve, False)[0]

                modlist.append((lmodel/candidate.err[band]).ravel()[mask_r])

                modarr = np.array(modlist).T

                amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

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

        print("fitting arc-like foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value))

        count += 1
        sampler.run_mcmc(start, nsamp)

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # fixes the amplitude to the best fit value, in each band
        for band in candidate.bands:

            modlist = []
            fixedcomps = 0.*candidate.sci[band]
            for l in scalefreemodels[band]:
                fixedcomps += l

            modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

            arc['model'][band].amp = 1.
            arc['model'][band].setPars()
            lmodel = convolve.convolve(arc['model'][band].pixeval(candidate.X, candidate.Y), \
                                        arc['model'][band].convolve, False)[0]
            arc['unitampmodel'][band] = lmodel

            modlist.append((lmodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

            if amps[0] == 0.:
                amps[0] = 1.
            arc['scalefreemodel'][band] = amps[1]/amps[0]*lmodel
            scalefreemodels[band].append(amps[1]/amps[0]*lmodel)

            if amps[1] <= 0.:
                mag = 99.
            else:
                mag = arc['model'][band].Mag(candidate.zp[band]) - 2.5*np.log10(amps[1]/amps[0])

            arc['scalefreemags'][band] = mag

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        lmodel = 0.*candidate.sci[band]
        for i in range(count):
            lmodel += scalefreemodels[band][i]

        fitmodel = np.atleast_2d((lmodel/candidate.err[band]).ravel()[mask_r]).T

        amps, chi = nnls(fitmodel, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        logp = -0.5*chi

        resid = candidate.sci[band].copy()
        resid -= lmodel*amps[0]

        candidate.foreground_model[band] = []
        for i in range(count):
            candidate.foreground_model[band].append(amps[0]*scalefreemodels[band][i])


def fit_new_foregrounds(candidate, foreground_model, light_model, rmax=30., nsamp=200):

    scalefreemodels = {}

    for band in candidate.bands:
        #foreground_model.amps[band] = []
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]
        scalefreemodels[band] = [lmodel]

        for comp in foreground_model.components:
            if comp['dofit'] == True:
                scalefreemodels[band].append(comp['scalefreemodel'][band])

    count = len(scalefreemodels[band])

    for obj in foreground_model.new_foregrounds:

        tmp_mask = np.ones(candidate.imshape, dtype=int)
        tmp_mask[candidate.R > rmax] = 0
        mask_r = (tmp_mask > 0).ravel()

        pars = obj['pars']

        obj['scalefreemodel'] = {}
        obj['scalefreemags'] = {}
        obj['unitampmodel'] = {}

        npars = len(pars)

        bounds = []

        for par in pars:
            bounds.append((par.lower, par.upper))

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

            for band in candidate.fitband:

                modlist = []
                fixedcomps = 0.*candidate.sci[band]
                for l in scalefreemodels[band]:
                    fixedcomps += l

                modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

                obj['model'][band].amp = 1.
                obj['model'][band].setPars()
                lmodel = convolve.convolve(obj['model'][band].pixeval(candidate.X, candidate.Y), \
                                            obj['model'][band].convolve, False)[0]

                modlist.append((lmodel/candidate.err[band]).ravel()[mask_r])

                modarr = np.array(modlist).T

                amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

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

        print("fitting arc-like foreground no. %d at x: %2.1f y: %2.1f"%(count, pars[0].value, pars[1].value))

        count += 1
        sampler.run_mcmc(start, nsamp)

        ML = sampler.flatlnprobability.argmax()

        for j in range(0, npars):
            pars[j].value = sampler.flatchain[ML, j]

        # fixes the amplitude to the best fit value, in each band
        for band in candidate.bands:

            modlist = []
            fixedcomps = 0.*candidate.sci[band]
            for l in scalefreemodels[band]:
                fixedcomps += l

            modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

            obj['model'][band].amp = 1.
            obj['model'][band].setPars()
            lmodel = convolve.convolve(obj['model'][band].pixeval(candidate.X, candidate.Y), \
                                        obj['model'][band].convolve, False)[0]
            obj['unitampmodel'][band] = lmodel

            modlist.append((lmodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

            if amps[0] == 0.:
                amps[0] = 1.
            obj['scalefreemodel'][band] = amps[1]/amps[0]*lmodel
            scalefreemodels[band].append(amps[1]/amps[0]*lmodel)

            if amps[1] <= 0.:
                mag = 99.
            else:
                mag = obj['model'][band].Mag(candidate.zp[band]) - 2.5*np.log10(amps[1]/amps[0])

            obj['scalefreemags'][band] = mag

    # removes the best-fit i-band model from all bands and saves the residuals
    for band in candidate.bands:

        lmodel = 0.*candidate.sci[band]
        for i in range(count):
            lmodel += scalefreemodels[band][i]

        fitmodel = np.atleast_2d((lmodel/candidate.err[band]).ravel()[mask_r]).T

        amps, chi = nnls(fitmodel, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        logp = -0.5*chi

        resid = candidate.sci[band].copy()
        resid -= lmodel*amps[0]

        candidate.foreground_model[band] = []
        for i in range(count):
            candidate.foreground_model[band].append(amps[0]*scalefreemodels[band][i])

def fit_lens_freeamps(candidate, lens_model, light_model, foreground_model, image_set, rmax=30., nsamp=200):

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
            lens_model.pa.lower = lens_model.pa.value - 30.
            lens_model.pa.upper = lens_model.pa.value + 30.
        else:
            lens_model.q.lower = 0.5
    else:
        lens_model.q.value = 1./candidate.light_q
        lens_model.pa.value = candidate.light_pa + 90.
        if candidate.light_q > 1./0.7:
            lens_model.pa.lower = lens_model.pa.value - 30.
            lens_model.pa.upper = lens_model.pa.value + 30.
        else:
            lens_model.q.lower = 0.5

    lens_model.rein.upper = image_set['furthest_arc']

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
        bounds.append((par.lower, par.upper))

    nwalkers = 50

    foregrounds = {}
    for band in candidate.bands:
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]

        foregrounds[band]  = [lmodel]
        for comp in foreground_model.components:
            if comp['dofit'] == True:
                foregrounds[band].append(comp['unitampmodel'][band])

        for arc in foreground_model.bad_arcs:
            foregrounds[band].append(arc['unitampmodel'][band])

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

        lens_model.lens.setPars()
        xl, yl = pylens.getDeflections(lens_model.lens, [candidate.X, candidate.Y])

        for band in candidate.fitband:

            modlist = []
            for l in foregrounds[band]:
                modlist.append((l/candidate.err[band]).ravel()[mask_r])

            lens_model.source[band].setPars()
            lens_model.source[band].amp = 1.
            smodel = convolve.convolve(lens_model.source[band].pixeval(xl, yl), lens_model.source[band].convolve, \
                                       False)[0]

            modlist.append((smodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            if np.isnan(modarr).any():
                return -1e300

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

            logp = -0.5*chi

            if logp != logp:
                return -np.inf
            sumlogp += logp

        return sumlogp

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

    print("fitting lens model...")

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

    lens_model.lens.setPars()
    xl, yl = pylens.getDeflections(lens_model.lens, [candidate.X, candidate.Y])

    for band in candidate.bands:

        modlist = []
        for l in foregrounds[band]:
            modlist.append((l/candidate.err[band]).ravel()[mask_r])

        lens_model.source[band].setPars()
        lens_model.source[band].amp = 1.
        smodel = convolve.convolve(lens_model.source[band].pixeval(xl, yl), lens_model.source[band].convolve, \
                                   False)[0]

        modlist.append((smodel/candidate.err[band]).ravel()[mask_r])

        modarr = np.array(modlist).T

        amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        candidate.lensfit_model[band] = []
        for i in range(len(foregrounds[band])):
            candidate.lensfit_model[band].append(amps[i]*foregrounds[band][i])

        candidate.lensfit_model[band].append(amps[-1]*smodel)

        if band in candidate.fitband:
            resid = candidate.sci[band].copy()
            for mimg in candidate.lensfit_model[band]:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.lensfit_chi2 = chi2
    candidate.lensfit_mask = mask


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
            lens_model.pa.lower = lens_model.pa.value - 30.
            lens_model.pa.upper = lens_model.pa.value + 30.
        else:
            lens_model.q.lower = 0.5
    else:
        lens_model.q.value = 1./candidate.light_q
        lens_model.pa.value = candidate.light_pa + 90.
        if candidate.light_q > 1./0.7:
            lens_model.pa.lower = lens_model.pa.value - 30.
            lens_model.pa.upper = lens_model.pa.value + 30.
        else:
            lens_model.q.lower = 0.5

    lens_model.rein.upper = 1.5*image_set['furthest_arc']
    lens_model.rein.lower = 0.3*image_set['mean_arc_dist']

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
        bounds.append((par.lower, par.upper))

    nwalkers = 50

    foregrounds = {}
    scalefreemags = {}
    for band in candidate.bands:
        light_model.model[band].setPars()
        light_model.model[band].amp = 1.
        lmodel = convolve.convolve(light_model.model[band].pixeval(candidate.X, candidate.Y), \
                                            light_model.model[band].convolve, False)[0]

        foregrounds[band]  = [lmodel.copy()]
        scalefreemags[band] = [light_model.model[band].Mag(candidate.zp[band])]

        for comp in foreground_model.components:
            if comp['dofit'] == True:
                foregrounds[band].append(comp['scalefreemodel'][band])
                scalefreemags[band].append(comp['scalefreemags'][band])

        for arc in foreground_model.bad_arcs:
            foregrounds[band].append(arc['scalefreemodel'][band])
            scalefreemags[band].append(arc['scalefreemags'][band])

        for obj in foreground_model.new_foregrounds:
            foregrounds[band].append(obj['scalefreemodel'][band])
            scalefreemags[band].append(obj['scalefreemags'][band])

    nfitband = len(candidate.fitband)

    placeholder_mags = np.inf*np.ones((2, nfitband))

    print('fitting %d foregrounds, including 1 lens and %d arc-like objects'%(len(foregrounds[band]), len(foreground_model.bad_arcs)))

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
            return -np.inf, placeholder_mags

        for j in range(0, npars):
            pars[j].value = allpars[j]
        sumlogp = 0.

        lens_model.lens.setPars()
        xl, yl = pylens.getDeflections(lens_model.lens, [candidate.X, candidate.Y])

        allmags = np.zeros((2, nfitband))

        for n in range(nfitband):

            band = candidate.fitband[n]

            modlist = []
            fixedcomps = 0.*candidate.sci[band]
            for l in foregrounds[band]:
                fixedcomps += l

            modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

            lens_model.source[band].setPars()
            lens_model.source[band].amp = 1.
            smodel = convolve.convolve(lens_model.source[band].pixeval(xl, yl), lens_model.source[band].convolve, \
                                       False)[0]

            modlist.append((smodel/candidate.err[band]).ravel()[mask_r])

            modarr = np.array(modlist).T

            if np.isnan(modarr).any():
                return -1e300, placeholder_mags

            amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

            logp = -0.5*chi

            if logp != logp:
                return -np.inf, placeholder_mags
            sumlogp += logp

            allmags[0, n] = scalefreemags[band][0] - 2.5*np.log10(amps[0])
            allmags[1, n] = lens_model.source[band].Mag(candidate.zp[band]) - 2.5*np.log10(amps[1])

        return sumlogp, allmags

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

    print("fitting lens model...")

    sampler.run_mcmc(start, nsamp)

    pars_sample = {}
    pars_sample['lens_rein'] = sampler.chain[:, :, 0]
    pars_sample['lens_q'] = sampler.chain[:, :, 1]
    pars_sample['lens_pa'] = sampler.chain[:, :, 2]
    pars_sample['source_x'] = sampler.chain[:, :, 3]
    pars_sample['source_y'] = sampler.chain[:, :, 4]
    pars_sample['source_re'] = sampler.chain[:, :, 5]

    blobarr = np.array(sampler.blobs)
    print(blobarr.shape)
    magschain = blobarr.reshape((nsamp, nwalkers, 2, nfitband))

    for n in range(nfitband):
        band = candidate.fitband[n]
        pars_sample['lens_%s_mag'%band] = magschain[:, :, 0, n].T
        pars_sample['source_%s_mag'%band] = magschain[:, :, 1, n].T

    candidate.lens_pars_sample = pars_sample

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

    lens_model.lens.setPars()
    xl, yl = pylens.getDeflections(lens_model.lens, [candidate.X, candidate.Y])

    candidate.lensfit_model = {}

    for band in candidate.bands:

        modlist = []
        fixedcomps = 0.*candidate.sci[band]
        for l in foregrounds[band]:
            fixedcomps += l

        modlist.append((fixedcomps/candidate.err[band]).ravel()[mask_r])

        lens_model.source[band].setPars()
        lens_model.source[band].amp = 1.
        smodel = convolve.convolve(lens_model.source[band].pixeval(xl, yl), lens_model.source[band].convolve, \
                                   False)[0]

        modlist.append((smodel/candidate.err[band]).ravel()[mask_r])

        modarr = np.array(modlist).T

        amps, chi = nnls(modarr, (candidate.sci[band]/candidate.err[band]).ravel()[mask_r])

        candidate.lensfit_model[band] = []
        for l in foregrounds[band]:
            candidate.lensfit_model[band].append(amps[0]*l)

        candidate.lensfit_mags[band] = []
        for mag in scalefreemags[band]:
            candidate.lensfit_mags[band].append(mag - 2.5*np.log10(amps[0]))

        candidate.lensfit_model[band].append(amps[1]*smodel)
        if amps[1] <= 0.:
            smag = 99.
        else:
            smag = lens_model.source[band].Mag(candidate.zp[band])

        candidate.lensfit_mags[band].append(smag - 2.5*np.log10(amps[1]))

        if band in candidate.fitband:
            resid = candidate.sci[band].copy()
            for mimg in candidate.lensfit_model[band]:
                resid -= mimg
            chi2 += ((resid/candidate.err[band])**2)[mask > 0].sum()

    candidate.lensfit_chi2 = chi2
    candidate.lensfit_mask = mask

