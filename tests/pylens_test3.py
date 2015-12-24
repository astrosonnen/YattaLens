# first fits only the arc, then uses the best lens model as starting point to fit the lens light as well

import pymc
import pyfits
import numpy as np
from photofit import convolve, models, indexTricks as iT
from pylens import pylens, MassModels
import pylab
from plotters import cornerplot


lensname = 'SL2SJ021411-040502'
dpath = '../photofit/example/'
imgfile = dpath+lensname+'_g_one_sersic_mcmc.fits'
sigfile = dpath+lensname+'_g_rms.fits'
psffile = dpath+lensname+'_g_psf.fits'
maskfile = dpath+lensname+'_mask.fits'

zp = 30.
Nsamp = 11000
burnin = 1000

sigma = pyfits.open(sigfile)[0].data.copy()
psf = pyfits.open(psffile)[0].data.copy()
image = pyfits.open(imgfile)[2].data.copy()
mask = pyfits.open(maskfile)[0].data.copy()
mask = np.logical_not(mask==0)

sigma_r = sigma.ravel()
image_r = image.ravel()
mask_r = mask.ravel()



#lens parameters
x0 = 19.907
y0 = 18.750
#x = pymc.Normal('x', mu=x0, tau=1./0.1**2, value=x0)
#y = pymc.Normal('y', mu=y0, tau=1./0.1**2, value=y0)
re = pymc.Uniform('re', lower=0., upper=20., value=4.)
pa = pymc.Uniform('pa', lower=-90., upper=180., value=0.)
q = pymc.Uniform('q', lower=0.3, upper=1., value=0.8)

#source parameters
sx = pymc.Uniform('sx', lower=15., upper=25., value=23.)
sy = pymc.Uniform('sy', lower=15., upper=25., value=20.)
sp = pymc.Uniform('sp', lower=-90., upper=180., value=0.)
sq = pymc.Uniform('sq', lower=0.3, upper=1., value=0.8)
sr = pymc.Uniform('sr', lower=1., upper=20., value=5.)

svar = {'x': 3, 'y': 4, 'pa': 5, 'q': 6, 're': 7}
sconst = {'amp':1., 'n':1.}
pars = [re, pa, q, sx, sy, sp, sq, sr]
guess = [10., 0., 0.8, 23., 20., 0., 0.8, 5.]

cov = [0.1, 1., 0.01, 1., 1., 1., 0.01, 1.]

#defines the lnes model
lens = MassModels.PowerLaw('lens', {'x':x0, 'y':y0, 'b':re, 'q':q, 'pa':pa, 'eta':1.})
#source = models.Sersic('src', {'x':sx, 'y':sy, 're':sr, 'q':sq, 'pa':sp, 'n':1.})
source = models.Sersic('source', svar, sconst)
source.convolve = convolve.convolve(image, psf)[1]

Y, X = iT.coords(image.shape)
xc = X.mean()
yc = Y.mean()
dists = ((Y - yc)**2 + (X - xc)**2)**0.5
newmask = np.ones(mask.shape)
newmask[dists < 3.] = 0.
mask = np.logical_not(newmask==0)
mask_r = mask.ravel()

logp, mag, modelimg = pylens.getModel_sourceonly(lens, source, guess, image, sigma, mask, X, Y, zp=zp, returnImg=True)

"""
cmap = 'binary'
pylab.subplot(1, 2, 1)
pylab.imshow(image*mask, cmap=cmap)

pylab.subplot(1, 2, 2)
pylab.imshow(modelimg*mask, cmap=cmap)

#pylab.subplot(2, 2, 3)
#pylab.imshow(mask, cmap=cmap)
pylab.show()
"""



simage = (image_r/sigma_r)[mask_r]

@pymc.deterministic(trace=False)
def logpAndMags(p=pars):
    logp, mags = pylens.getModel_sourceonly(lens, source, p, image, sigma, mask, X, Y, zp=zp)
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

"""
cornerplot(cp, color='r')
pylab.show()

pylab.subplot(2, 1, 1)
pylab.plot(trace['logp'])
pylab.subplot(2, 1, 2)
pylab.plot(trace['re'])
pylab.show()
"""


ML = trace['logp'].argmax()
mlpars = []
for par in pars:
    mlval = trace[str(par)][ML]
    print str(par), mlval
    mlpars.append(mlval)

logp, mag, mimage = pylens.getModel_sourceonly(lens, source, mlpars, image, sigma, mask, X, Y, returnImg=True)

"""
pylab.subplot(1, 2, 1)
pylab.imshow(image)

pylab.subplot(1, 2, 2)
pylab.imshow(mimage)

pylab.show()
"""


#now uses the best fit lens model to try and fit lens light and source simultaneously

lensname = 'SL2SJ021411-040502'
dpath = '../photofit/example/'

bands = ['i', 'g']
bands = ['g']

zps = {'i': 30., 'g': 30.}

Nsamp = 10000
burnin = 0

sigmas = {}
images = {}
psfs = {}

for band in bands:
    images[band] = pyfits.open(dpath+lensname+'_%s_sci.fits'%band)[0].data.copy()
    sigmas[band] = pyfits.open(dpath+lensname+'_%s_rms.fits'%band)[0].data.copy()
    psfs[band] = pyfits.open(dpath+lensname+'_%s_psf.fits'%band)[0].data.copy()


#lens parameters
x = pymc.Uniform('x', lower=18., upper=22., value=x0)
y = pymc.Uniform('y', lower=18., upper=22., value=y0)
rein = pymc.Uniform('rein', lower=0., upper=20., value=mlpars[0])
pa = pymc.Uniform('pa', lower=-90., upper=180., value=mlpars[1])
q = pymc.Uniform('q', lower=0.3, upper=1., value=mlpars[2])

#light parameters
reff = pymc.Uniform('reff', lower=1., upper=20., value=5.)
pas = pymc.Uniform('pas', lower=-90., upper=180., value=mlpars[1])
qs = pymc.Uniform('qs', lower=0.3, upper=1., value=0.8)


#source parameters
sx = pymc.Uniform('sx', lower=15., upper=25., value=mlpars[3])
sy = pymc.Uniform('sy', lower=15., upper=25., value=mlpars[4])
sp = pymc.Uniform('sp', lower=-90., upper=180., value=mlpars[5])
sq = pymc.Uniform('sq', lower=0.3, upper=1., value=mlpars[6])
sr = pymc.Uniform('sr', lower=1., upper=20., value=mlpars[7])


pars = [x, y, rein, pa, q, reff, pas, qs, sx, sy, sp, sq, sr]
cov = [0.01, 0.01, 0.01, 1., 0.01, 1., 1., 0.01, 0.01, 0.01, 1., 0.01, 0.1]

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
lconst = {'amp':1., 'n':4.}

svar = {'x': 8, 'y': 9, 'pa': 10, 'q': 11, 're': 12}
sconst = {'amp':1., 'n':1.}

#defines the lnes model
lens = MassModels.PowerLaw('lens', {'x':x, 'y':y, 'b':rein, 'q':q, 'pa':pa, 'eta':1.})
#source = models.Sersic('src', {'x':sx, 'y':sy, 're':sr, 'q':sq, 'pa':sp, 'n':1.})

lights = {}
sources = {}
scaleguess = {'i': 2., 'g':0.8}
scalepars = []
scalecov = []

for band in bands:
    light = models.Sersic('LensLight', lvar, lconst)
    source = models.Sersic('source', svar, sconst)

    light.convolve = convolve.convolve(images[band], psfs[band])[1]
    source.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light
    sources[band] = source

    scalepars.append(pymc.Uniform('dm_%s'%band, lower=0., upper=5., value=scaleguess[band]))
    scalecov.append(0.1)

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
"""
i = 0
for par in pars:
    M.use_step_method(pymc.Metropolis, par, proposal_sd=cov[i])
    i += 1
i = 0
for band in bands:
    M.use_step_method(pymc.Metropolis, scalepars[i], proposal_sd=scalecov[i])
"""
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





