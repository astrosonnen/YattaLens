# fits the arc in the g band



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
Nsamp = 110000
burnin = 10000

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
lenses = [lens]
models = [source]

nmod = len(models) #number of components (either lens or source)

Y, X = iT.coords(image.shape)
xc = X.mean()
yc = Y.mean()
dists = ((Y - yc)**2 + (X - xc)**2)**0.5
newmask = np.ones(mask.shape)
newmask[dists < 3.] = 0.
mask = np.logical_not(newmask==0)
mask_r = mask.ravel()

logp, mag, modelimg = pylens.getModel_sourceonly(lens, source, guess, image, sigma, mask, X, Y, zp=zp, returnImg=True)

cmap = 'binary'
pylab.subplot(1, 2, 1)
pylab.imshow(image*mask, cmap=cmap)

pylab.subplot(1, 2, 2)
pylab.imshow(modelimg*mask, cmap=cmap)

#pylab.subplot(2, 2, 3)
#pylab.imshow(mask, cmap=cmap)
pylab.show()



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

cornerplot(cp, color='r')
pylab.show()

pylab.subplot(2, 1, 1)
pylab.plot(trace['logp'])
pylab.subplot(2, 1, 2)
pylab.plot(trace['re'])
pylab.show()


ML = trace['logp'].argmax()
mlpars = []
for par in pars:
    mlval = trace[str(par)][ML]
    print str(par), mlval
    mlpars.append(mlval)

logp, mag, mimage = pylens.getModel_sourceonly(lens, source, mlpars, image, sigma, mask, X, Y, returnImg=True)

pylab.subplot(1, 2, 1)
pylab.imshow(image)

pylab.subplot(1, 2, 2)
pylab.imshow(mimage)

pylab.show()






