# fits lens light and arc in one band, using emcee

import pymc
import emcee
import pyfits
import numpy as np
from photofit import convolve, models, indexTricks as iT
from pylens import pylens, MassModels
import pylab
#from plotters import cornerplot


lensname = 'SL2SJ021411-040502'
dpath = '../photofit/example/'

bands = ['i', 'g']
bands = ['g']

zps = {'i': 30., 'g': 30.}

Nsamp = 100000
burnin = 0

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
cov = [0.01, 0.01, 0.1, 1., 0.01, 1., 1., 0.01, 0.01, 0.01, 1., 0.01, 1.]

guess = []
for par in pars:
    guess.append(par.value)

lvar = {'x': 0, 'y': 1, 're': 5, 'pa': 6, 'q':7}
lconst = {'amp':1., 'n':1.}

svar = {'x': 8, 'y': 9, 'pa': 10, 'q': 11, 're': 12}
sconst = {'amp':1., 'n':1.}

#defines the lnes model
lens = MassModels.PowerLaw('lens', {'x':x, 'y':y, 'b':rein, 'q':q, 'pa':pa, 'eta':1.})
#source = models.Sersic('src', {'x':sx, 'y':sy, 're':sr, 'q':sq, 'pa':sp, 'n':1.})

lights = {}
sources = {}
scaleguess = {'i': 2., 'g':-0.5}
scalepars = []
scalecov = []

for band in bands:
    light = models.Sersic('LensLight', lvar, lconst)
    source = models.Sersic('source', svar, sconst)

    light.convolve = convolve.convolve(images[band], psfs[band])[1]
    source.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light
    sources[band] = source

    scalepars.append(pymc.Uniform('dm_%s'%band, lower=-2., upper=5., value=scaleguess[band]))
    scalecov.append(0.01)

    Y, X = iT.coords(images[band].shape)

    mask = np.zeros(images[band].shape) == 0

    logp, mags, modelimg = pylens.getModel(lens, light, source, guess, scaleguess[band], images[band], sigmas[band], \
                                           mask, X, Y, zp=zps[band], returnImg=True)

    cmap = 'binary'
    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band], cmap=cmap)

    pylab.subplot(1, 2, 2)
    pylab.imshow(modelimg, cmap=cmap)

    pylab.show()


@pymc.deterministic(trace=False)
def logpAndMags(p=pars, s=scalepars):
    sumlogp = 0.
    magslist = []
    i = 0
    for band in bands:
        logp, mags = pylens.getModel(lens, lights[band], sources[band], p, s[i], images[band], sigmas[band], mask, \
                                     X, Y, zp=zps[band])
        if logp != logp:
            return -1e300, []
        sumlogp += logp
        magslist.append(mags)
        i += 1

    return sumlogp, magslist

@pymc.deterministic
def lp(lpAM=logpAndMags):
    return lpAM[0]

@pymc.deterministic(name='Mags')
def Mags(lpAM=logpAndMags):
    return lpAM[1]

@pymc.stochastic(observed=True, name='logp')
def logpCost(value=0., p=pars, s=scalepars):
    if lp != lp:
        return -1e300
    return lp

print "Sampling"

M = pymc.MCMC(pars+scalepars+[lp, Mags])
M.use_step_method(pymc.AdaptiveMetropolis, pars+scalepars, cov=np.diag(cov+scalecov))
M.isample(Nsamp, burnin)

trace = {}
cp = []
for par in pars+scalepars:
    trace[str(par)] = M.trace(par)[:]
    cp.append({'data':trace[str(par)], 'label':str(par)})

trace['logp'] = M.trace('lp')[:]

pylab.plot(trace['logp'])
pylab.show()
pylab.plot(trace['rein'])
pylab.show()

#cornerplot(cp, color='r')
#pylab.show()


ML = trace['logp'].argmax()
print ML
mlpars = []
mlscalepars = {}
for par in pars:
    mlval = trace[str(par)][ML]
    print str(par), mlval
    mlpars.append(mlval)
    par.value = mlval

for band in bands:
    parname = 'dm_%s'%band
    mlval = trace[parname][ML]
    print parname, mlval
    mlscalepars[band] = mlval

for band in bands:
    logp, mags, mimage = pylens.getModel(lens, lights[band], sources[band], mlpars, mlscalepars[band], images[band], \
                                         sigmas[band], mask, X, Y, returnImg=True)

    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band])

    pylab.subplot(1, 2, 2)
    pylab.imshow(mimage)

    pylab.show()






