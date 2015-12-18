import pymc
import pyfits
import numpy as np
from photofit import convolve, models, indexTricks as iT
from photofit.SampleOpt import AMAOpt, Sampler
from pylens import pylens, MassModels
from scipy.optimize import fmin_slsqp
import pylab
from plotters import cornerplot


lensname = 'SL2SJ021411-040502'
dpath = '../photofit/example/'

bands = ['i', 'g']

zps = {'i': 30., 'g': 30.}

Nsamp = 110000
burnin = 10000

sigmas = {}
images = {}
psfs = {}

for band in bands:
    images[band] = pyfits.open(dpath+lensname+'_%s_one_sersic_mcmc.fits'%band)[0].data.copy()
    sigmas[band] = pyfits.open(dpath+lensname+'_%s_rms.fits'%band)[0].data.copy()
    psfs[band] = pyfits.open(dpath+lensname+'_%s_psf.fits'%band)[0].data.copy()


#lens parameters
x0 = 19.907
y0 = 18.750
#x = pymc.Normal('x', mu=x0, tau=1./0.1**2, value=x0)
#y = pymc.Normal('y', mu=y0, tau=1./0.1**2, value=y0)
x = pymc.Uniform('x', lower=18., upper=22., value=x0)
y = pymc.Uniform('x', lower=18., upper=22., value=y0)
rein = pymc.Uniform('rein', lower=0., upper=20., value=4.)
pa = pymc.Uniform('pa', lower=-90., upper=180., value=0.)
q = pymc.Uniform('q', lower=0.3, upper=1., value=0.8)

#light parameters
reff = pymc.Uniform('reff', lower=1., upper=20., value=5.)
pas = pymc.Uniform('pas', lower=-90., upper=180., value=0.)
qs = pymc.Uniform('qs', lower=0.3, upper=1., value=0.8)


#source parameters
sx = pymc.Uniform('sx', lower=15., upper=25., value=23.)
sy = pymc.Uniform('sy', lower=15., upper=25., value=20.)
sp = pymc.Uniform('sp', lower=-90., upper=180., value=0.)
sq = pymc.Uniform('sq', lower=0.3, upper=1., value=0.8)
sr = pymc.Uniform('sr', lower=1., upper=20., value=5.)


pars = [x, y, rein, pa, q, reff, pas, qs, sx, sy, sp, sq, sr]
cov = [0.01, 0.01, 0.1, 1., 0.01, 1., 1., 0.01, 1., 1., 1., 0.01, 1.]

guess = []
for par in pars:
    guess.append(par.value)

print guess

"""
# filter mismatch parameters
for band in bands[1:]:
    pars.append(pymc.Uniform('dx_%s'%band, lower=-1., upper=1., value=0.))
    pars.append(pymc.Uniform('dy_%s'%band, lower=-1., upper=1., value=0.))
    cov.append(0.01)
    cov.append(0.01)
"""

lvar = {'x': 0, 'y': 1, 're': 5, 'pa': 6, 'q':7}
lconst = {'amp':1., 'n':1.}

svar = {'x': 8, 'y': 9, 'pa': 10, 'q': 11, 're': 12}
sconst = {'amp':1., 'n':1.}

#defines the lnes model
lens = MassModels.PowerLaw('lens', {'x':x, 'y':y, 'b':rein, 'q':q, 'pa':pa, 'eta':1.})
#source = models.Sersic('src', {'x':sx, 'y':sy, 're':sr, 'q':sq, 'pa':sp, 'n':1.})

lights = {}
sources = {}

for band in bands:
    light = models.Sersic('LensLight', lvar, lconst)
    source = models.Sersic('source', svar, sconst)

    light.convolve = convolve.convolve(images[band], psfs[band])[1]
    source.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light
    sources[band] = source

    Y, X = iT.coords(images[band].shape)

    mask = np.zeros(images[band].shape) == 0

    logp, mag, modelimg = pylens.getModel(lens, source, guess, images[band], sigmas[band], mask, X, Y, zp=zps[band], lenslight=light, returnImg=True)

    cmap = 'binary'
    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band], cmap=cmap)

    pylab.subplot(1, 2, 2)
    pylab.imshow(modelimg, cmap=cmap)

    pylab.show()

lenses = [lens]




simage = (image_r/sigma_r)[mask_r]

@pymc.deterministic(trace=False)
def logpAndMags(p=pars):
    logp, mags = pylens.getModel(lens, source, p, image, sigma, mask, X, Y, zp=zp)
    if logp != logp:
        logp = -1e300
    return logp, mags

@pymc.deterministic
def lp(lpAM=logpAndMags):
    return lpAM[0]

@pymc.deterministic(name='Mags')
def Mags(lpAM=logpAndMags):
    return lpAM[1]

@pymc.stochastic(observed=True, name='logp')
def logpCost(value=0., p=pars):
    #if lp != lp:
    #    return -1300
    #else:
    return lp

print "Sampling"

M = pymc.MCMC(pars+[lp, Mags])
M.use_step_method(pymc.AdaptiveMetropolis, pars, cov=np.diag(cov))
M.isample(Nsamp, burnin)

trace = {}
cp = []
for par in pars:
    trace[str(par)] = M.trace(par)[:]
    cp.append({'data':trace[str(par)], 'label':str(par)})

trace['logp'] = M.trace('lp')[:]

cornerplot(cp, color='r')
pylab.show()


ML = trace['logp'].argmax()
mlpars = []
for par in pars:
    mlval = trace[str(par)][ML]
    print str(par), mlval
    mlpars.append(mlval)

logp, mag, mimage = pylens.getModel(lens, source, mlpars, image, sigma, mask, X, Y, returnImg=True)

pylab.subplot(1, 2, 1)
pylab.imshow(image)

pylab.subplot(1, 2, 2)
pylab.imshow(mimage)

pylab.show()






