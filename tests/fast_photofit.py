import numpy as np,time
from math import log10
import pymc, emcee
import sys
import pyfits
from photofit import imageFit, indexTricks as iT, convolve
from pylens import pylens, SBModels as models
import pickle
import pylab
from scipy.stats import truncnorm


bands0 = ['i']

bands = ['u', 'g', 'r', 'i', 'z']
zps = {'u':30., 'g':30., 'r':30., 'i':30., 'z':30.}

dpath = '../photofit/example/'
lensname = 'SL2SJ021411-040502'

x0 = 20.
y0 = 20.
dx = 2.
dy = 2.

x = pymc.Uniform('x', lower=x0 - dx, upper=x0 + dx, value=x0)
y = pymc.Uniform('y', lower=y0 - dy, upper=y0 + dy, value=y0)
pa = pymc.Uniform('pa', lower=-180., upper=180., value=0.)
q = pymc.Uniform('q', lower=0.3, upper=1., value=0.7)
reff = pymc.Uniform('re', lower=3., upper=30., value=10.)

pars = [x, y, pa, q, reff]
npars = len(pars)

nwalkers = 6*npars

nstep = 200

bounds = []

for par in pars:
    bounds.append((par.parents['lower'], par.parents['upper']))

start = []
for i in range(nwalkers):
    urand = np.random.rand(npars)
    p0 = 0.*urand
    for j in range(npars):
        if j == 2:
            p0[j] = urand[j]*180. - 90.
        else:
            p0[j] = urand[j]*(bounds[j][1] - bounds[j][0]) + bounds[j][0]

    start.append(p0)


images = {}
sigmas = {}
psfs = {}
for band in bands:
    images[band] = pyfits.open(dpath+lensname+'_%s_sci.fits'%band)[0].data.copy()
    sigmas[band] = pyfits.open(dpath+lensname+'_%s_rms.fits'%band)[0].data.copy()
    psfs[band] = pyfits.open(dpath+lensname+'_%s_psf.fits'%band)[0].data.copy()

Y, X = iT.coords(images[bands[0]].shape)
mask = np.ones(images[bands[0]].shape, dtype=bool)
mask_r = mask.ravel()

lights = {}
for band in bands:
    light = models.Sersic('LensLight', {'x': x, 'y': y, 're': reff, 'q': q, 'pa': pa, 'n': 4.})

    light.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light



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

    for band in bands0:
        logp, mag = pylens.getModel_lightonly(lights[band], images[band], sigmas[band], X, Y, zp=zps[band], mask=mask_r)

        if logp != logp:
            return -np.inf

        sumlogp += logp

    return sumlogp


sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

sampler.run_mcmc(start, nstep)

ML = sampler.flatlnprobability.argmax()

steps = []
for j in range(npars):
    pars[j].value = sampler.flatchain[ML, j]
    print str(pars[j]), pars[j].value
    steps.append(sampler.flatchain[:, j].std())

for j in range(npars):
    pylab.plot(sampler.chain[:, :, j].T)
    pylab.title(str(pars[j]))
    pylab.show()

output = {'chain': sampler.chain, 'logp': sampler.lnprobability}

f = open('iband_fit.dat', 'w')
pickle.dump(output, f)
f.close()


# now fits all bands, adding Sersic index as free parameter

n = pymc.Uniform('n', lower=0., upper=8., value=4.)

for band in bands:
    light = models.Sersic('LensLight', {'x': x, 'y': y, 're': reff, 'q': q, 'pa': pa, 'n': n})
    light.convolve = convolve.convolve(images[band], psfs[band])[1]
    lights[band] = light

pars = [n] + pars
steps = [0.3] + steps
steps[3] = 5.
bounds = [(0., 8.)] + bounds
npars += 1

nwalkers += 6

start = []

for i in range(nwalkers):
    p0 = np.zeros(npars)
    for j in range(npars):
        a, b = (bounds[j][0] - pars[j].value)/steps[j], (bounds[j][1] - pars[j].value)/steps[j]
        p0[j] = truncnorm.rvs(a, b, size=1)*steps[j] + pars[j].value
    p0[npars - 1] += (p0[0] - 4.)*0.3*pars[npars - 1].value

    start.append(p0)



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

    for band in bands:
        logp, mag = pylens.getModel_lightonly(lights[band], images[band], sigmas[band], X, Y, zp=zps[band], mask=mask_r)

        if logp != logp:
            return -np.inf

        sumlogp += logp

    return sumlogp


sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

sampler.run_mcmc(start, nstep)

ML = sampler.flatlnprobability.argmax()

output = {'chain': sampler.chain, 'logp': sampler.lnprobability}

f = open('allbands_fit.dat', 'w')
pickle.dump(output, f)
f.close()

steps = []
for j in range(npars):
    pars[j].value = sampler.flatchain[ML, j]
    print str(pars[j]), pars[j].value
    steps.append(sampler.flatchain[:, j].std())

for j in range(npars):
    pylab.plot(sampler.chain[:, :, j].T)
    pylab.title(str(pars[j]))
    pylab.show()


