# fits only the arc, in two bands simultaneously



import pymc
import pyfits
import numpy as np
from photofit import convolve, models, indexTricks as iT
from pylens import pylens, MassModels
import pylab
#from plotters import cornerplot


lensname = 'SL2SJ021411-040502'
dpath = '../photofit/example/'

bands = ['i', 'g']

zps = {'i': 30., 'g': 30.}

Nsamp = 100000
burnin = 0

sigmas = {}
images = {}
psfs = {}

for band in bands:
    images[band] = pyfits.open(dpath+lensname+'_%s_one_sersic_mcmc.fits'%band)[2].data.copy()
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


#source parameters
sx = pymc.Uniform('sx', lower=15., upper=25., value=22.)
sy = pymc.Uniform('sy', lower=15., upper=25., value=19.)
sp = pymc.Uniform('sp', lower=-90., upper=180., value=0.)
sq = pymc.Uniform('sq', lower=0.3, upper=1., value=0.8)
sr = pymc.Uniform('sr', lower=1., upper=20., value=5.)


pars = [x, y, rein, pa, q, sx, sy, sp, sq, sr]
cov = [0.01, 0.01, 0.1, 1., 0.01, 0.1, 0.1, 1., 0.01, 1.]

guess = []
for par in pars:
    guess.append(par.value)

svar = {'x': 5, 'y': 6, 'pa': 7, 'q': 8, 're': 9}
sconst = {'amp':1., 'n':1.}

#defines the lnes model
lens = MassModels.PowerLaw('lens', {'x':x, 'y':y, 'b':rein, 'q':q, 'pa':pa, 'eta':1.})

sources = {}

for band in bands:
    source = models.Sersic('source', svar, sconst)

    source.convolve = convolve.convolve(images[band], psfs[band])[1]

    sources[band] = source

    Y, X = iT.coords(images[band].shape)

    mask = np.zeros(images[band].shape) == 0

    logp, mags, modelimg = pylens.getModel_sourceonly(lens, source, guess, images[band], sigmas[band], \
                                           mask, X, Y, zp=zps[band], returnImg=True)

    cmap = 'binary'
    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band], cmap=cmap)

    pylab.subplot(1, 2, 2)
    pylab.imshow(modelimg, cmap=cmap)

    pylab.show()


@pymc.deterministic(trace=False)
def logpAndMags(p=pars):
    sumlogp = 0.
    magslist = []
    i = 0
    for band in bands:
        logp, mags = pylens.getModel_sourceonly(lens, sources[band], p, images[band], sigmas[band], mask, \
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
def logpCost(value=0., p=pars):
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

pylab.subplot(2, 1, 1)
pylab.plot(trace['logp'])
pylab.subplot(2, 1, 2)
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
    logp, mags, mimage = pylens.getModel_sourceonly(lens, sources[band], mlpars, images[band], \
                                         sigmas[band], mask, X, Y, returnImg=True)

    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band])

    pylab.subplot(1, 2, 2)
    pylab.imshow(mimage)

    pylab.show()






