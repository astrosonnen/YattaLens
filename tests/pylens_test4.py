# fits lens light and arc in one band, using emcee

import pymc
import emcee
import pyfits
import numpy as np
from photofit import convolve, indexTricks as iT
from pylens import pylens, MassModels, SBModels as models
import pylab
#from plotters import cornerplot


lensname = 'SL2SJ021411-040502'
dpath = '../photofit/example/'

bands = ['i', 'g']
bands = ['g']

zps = {'i': 30., 'g': 30.}

Nsamp = 100
burnin = 0
nwalkers = 50

sigmas = {}
images = {}
psfs = {}

for band in bands:
    images[band] = pyfits.open(dpath+lensname+'_%s_sci.fits'%band)[0].data.copy()
    sigmas[band] = pyfits.open(dpath+lensname+'_%s_rms.fits'%band)[0].data.copy()
    psfs[band] = pyfits.open(dpath+lensname+'_%s_psf.fits'%band)[0].data.copy()


#lens parameters
x0 = 19.907
y0 = 20.
#x = pymc.Normal('x', mu=x0, tau=1./0.1**2, value=x0)
#y = pymc.Normal('y', mu=y0, tau=1./0.1**2, value=y0)
x = pymc.Uniform('x', lower=18., upper=22., value=x0)
y = pymc.Uniform('y', lower=18., upper=22., value=y0)
rein = pymc.Uniform('rein', lower=0., upper=20., value=6.)
pa = pymc.Uniform('pa', lower=-90., upper=180., value=0.)
q = pymc.Uniform('q', lower=0.3, upper=1., value=0.8)

#light parameters
reff = pymc.Uniform('reff', lower=1., upper=20., value=5.)
pas = pymc.Uniform('pas', lower=-90., upper=180., value=0.)
qs = pymc.Uniform('qs', lower=0.3, upper=1., value=0.8)


#source parameters
sx = pymc.Uniform('sx', lower=15., upper=25., value=22.)
sy = pymc.Uniform('sy', lower=15., upper=25., value=19.)
sp = pymc.Uniform('sp', lower=-90., upper=180., value=0.)
sq = pymc.Uniform('sq', lower=0.3, upper=1., value=0.8)
sr = pymc.Uniform('sr', lower=1., upper=20., value=5.)


pars = [x, y, rein, pa, q, reff, pas, qs, sx, sy, sp, sq, sr]
cov = [0.1, 0.1, 0.3, 1., 0.1, 1., 1., 0.1, 0.1, 0.1, 1., 0.1, 1.]

guess = []
for par in pars:
    guess.append(par.value)

#defines the lens model
lens = MassModels.PowerLaw('lens', {'x':x, 'y':y, 'b':rein, 'q':q, 'pa':pa, 'eta':1.})

lights = {}
sources = {}
scaleguess = {'i': 2., 'g':-0.5}

for band in bands:
    light = models.Sersic('LensLight', {'x':x, 'y':y, 're':reff, 'q':qs, 'pa':pas, 'n':4.})
    source = models.Sersic('source', {'x':sx, 'y':sy, 're':sr, 'q':sq, 'pa':sp, 'n':1.})

    light.convolve = convolve.convolve(images[band], psfs[band])[1]
    source.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light
    sources[band] = source

    pars.append(pymc.Uniform('dm_%s'%band, lower=-2., upper=5., value=scaleguess[band]))
    cov.append(0.1)
    guess.append(scaleguess[band])

    Y, X = iT.coords(images[band].shape)

    mask = np.zeros(images[band].shape) == 0

    logp, mags, modelimg = pylens.getModel_allpylens(lens, light, source, scaleguess[band], images[band], sigmas[band], \
                                           mask, X, Y, zp=zps[band], returnImg=True)

    cmap = 'binary'
    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band], cmap=cmap)

    pylab.subplot(1, 2, 2)
    pylab.imshow(modelimg, cmap=cmap)

    pylab.show()

npars = len(pars)


def logpfunc(allpars):
    structpars = allpars[:13]
    for j in range(0, 13):
        pars[j].value = structpars[j]
    sumlogp = 0.
    magslist = []
    i = 0
    for band in bands:

        logp, mags = pylens.getModel_allpylens(lens, lights[band], sources[band], allpars[13+i], images[band], sigmas[band], mask, \
                                     X, Y, zp=zps[band])
        if logp != logp:
            return -1e300, []
        sumlogp += logp
        magslist.append(mags)
        i += 1

    return sumlogp

sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

start = []
for i in range(nwalkers):
    tmp = []
    for j in range(0, npars):
        p0 = np.random.normal(guess[j], cov[j], 1)
        tmp.append(p0)
        pars[j].value = p0

    """
    for band in bands:
        logp, mags, modelimg = pylens.getModel_allpylens(lens, lights[band], sources[band], pars[-1].value, images[band], sigmas[band], \
                                               mask, X, Y, zp=zps[band], returnImg=True)

        cmap = 'binary'
        pylab.subplot(1, 2, 1)
        pylab.imshow(images[band], cmap=cmap)

        pylab.subplot(1, 2, 2)
        pylab.imshow(modelimg, cmap=cmap)

        pylab.show()
    """

    start.append(np.array(tmp).reshape(npars, ))


print "Sampling"

sampler.run_mcmc(start, Nsamp)

for i in range(npars):
    pylab.plot(sampler.flatchain[:, i])
    pylab.title(str(pars[i]))
    pylab.show()


