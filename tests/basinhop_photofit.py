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
from scipy.optimize import basinhopping


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
pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
q = pymc.Uniform('q', lower=0.3, upper=1., value=0.7)
reff = pymc.Uniform('re', lower=3., upper=30., value=10.)

pars = [x, y, pa, q, reff]
npars = len(pars)

nwalkers = 6*npars

nstep = 200

bounds = []
guess = []

for par in pars:
    bounds.append((par.parents['lower'], par.parents['upper']))
    guess.append(par.value)


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



"""
def logprior(allpars):
    for i in range(0, npars):
        if allpars[i] < bounds[i][0] or allpars[i] > bounds[i][1]:
            return -np.inf
    return 0.
"""


def logpfunc(scaledp):
    #lp = logprior(allpars)
    #if not np.isfinite(lp):
    #    return +np.inf

    p = scaledp*(barr[:,1] - barr[:,0]) + barr[:,0]
    for j in range(0, npars):
        pars[j].value = p[j]
    sumlogp = 0.

    for band in bands:
        logp, mag = pylens.getModel_lightonly(lights[band], images[band], sigmas[band], X, Y, zp=zps[band], mask=mask_r)

        if logp != logp:
            return -np.inf

        sumlogp += logp

    return -sumlogp


barr = np.array(bounds)
guess = np.array(guess)
scale_free_bounds = 0.*barr
scale_free_bounds[:,1] = 1.

scale_free_guess = (guess - barr[:,0])/(barr[:,1] - barr[:,0])

minimizer_kwargs = dict(method="L-BFGS-B", bounds=scale_free_bounds, tol=1.)

print 'optimizing...'

#res = basinhopping(logpfunc, scale_free_guess, stepsize=0.01, niter=500, minimizer_kwargs=minimizer_kwargs)
res = basinhopping(logpfunc, scale_free_guess, stepsize=0.1, niter=200, minimizer_kwargs=minimizer_kwargs, interval=30, T=3.)

MLpars = res.x*(barr[:,1] - barr[:,0]) + barr[:,0]

print MLpars, -res.fun

