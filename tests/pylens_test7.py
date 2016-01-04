# fits only light from the lens.

import pymc
import pyfits
import numpy as np
from photofit import convolve, indexTricks as iT
from pylens import pylens, MassModels, SBModels as models
import pylab
import pickle
from scipy.optimize import fmin_slsqp


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

mask = pyfits.open(dpath+lensname+'_mask.fits')[0].data.copy()
mask = mask==0

# light parameters
x0 = 19.907
y0 = 20.
x = pymc.Uniform('x', lower=18., upper=22., value=x0)
y = pymc.Uniform('y', lower=18., upper=22., value=y0)
reff = pymc.Uniform('reff', lower=5., upper=20., value=10.)
pas = pymc.Uniform('pas', lower=-90., upper=180., value=0.)
qs = pymc.Uniform('qs', lower=0.3, upper=1., value=0.8)
ns = pymc.Uniform('ns', lower=1., upper=8., value=4.)


pars = [x, y, reff, pas, qs, ns]
cov = [0.01, 0.01, 0.1, 1., 0.01, 0.01]

guess = []
for par in pars:
    guess.append(par.value)

print guess



def objf(x, lhs, rhs):
    return ((np.dot(lhs, x) - rhs)**3).sum()
def objdf(x, lhs, rhs):
    return np.dot(lhs.T, np.dot(lhs, x) - rhs)

def solve_linearproblem(profile, image, sigma, mask, X, Y, zp=30., returnImg=False):

    simage = ((image/sigma).ravel())[mask.ravel()]

    profile.setPars()
    profile.amp = 1.

    limg = profile.pixeval(X, Y)
    limg = convolve.convolve(limg, profile.convolve, False)[0]

    if np.isnan(limg).any():
        rimg = 0.*image
        rimg[limg!=limg] = 1.
        return 0., 0., rimg

    model = np.zeros((1, mask.sum()))

    lmodel = limg[mask].ravel()

    model[0] = lmodel

    norm = model[0].max()

    model[0] /= norm


    op = (model/sigma.ravel()[mask.ravel()]).T

    fit, chi = np.linalg.lstsq(op, simage)[:2]
    if (fit<0).any():
        return -1e300, 99.
        """
        sol = fit
        sol[sol<0] = 1e-11
        bounds = [(1e-11,1e11)]
        result = fmin_slsqp(objf, sol, bounds=bounds, full_output=1, fprime=objdf, acc=1e-19, iter=2000, args=(op.copy(), simage.copy()), iprint=0)
        fit, chi = result[:2]
        fit = np.asarray(fit)
        if (fit<1e-11).any():
            fit[fit<1e-11] = 1e-11
        """

    logp = -0.5*chi - np.log(sigma.ravel()[mask.ravel()]).sum()

    profile.amp = fit[0]/norm
    mag = light.Mag(zp)

    if returnImg:
        lsimg = profile.pixeval(X, Y)
        lsimg = convolve.convolve(lsimg, profile.convolve, False)[0]

        return logp, mag, lsimg

    else:
        return logp, mag


lights = {}

for band in bands:
    light = models.Sersic('LensLight', {'x': x, 'y': y, 're': reff, 'q': qs, 'pa': pas, 'n': ns})

    light.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light

    Y, X = iT.coords(images[band].shape)

    logp, mags, modelimg = solve_linearproblem(light, images[band], sigmas[band], mask, X, Y, zp=zps[band], returnImg=True)

    """
    cmap = 'binary'
    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band]*mask)#, cmap=cmap)

    pylab.subplot(1, 2, 2)
    pylab.imshow(modelimg*mask)#, cmap=cmap)

    pylab.show()
    """


@pymc.deterministic(trace=False)
def logpAndMags(p=pars):
    sumlogp = 0.
    magslist = []
    i = 0
    for band in bands:
        logp, mags = solve_linearproblem(lights[band], images[band], sigmas[band], mask, X, Y, zp=zps[band])
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
M.sample(Nsamp, burnin)

trace = {}
cp = []
for par in pars:
    trace[str(par)] = M.trace(par)[:]
    cp.append({'data':trace[str(par)], 'label':str(par)})

trace['logp'] = M.trace('lp')[:]

"""
pylab.plot(trace['logp'])
pylab.show()
pylab.plot(trace['reff'])
pylab.show()

cornerplot(cp, color='r')
pylab.show()
"""

f = open('test7.dat', 'w')
pickle.dump(trace, f)
f.close()

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
    logp, mags, mimage = solve_linearproblem(lights[band], images[band], sigmas[band], mask, X, Y, returnImg=True)

    pylab.subplot(1, 2, 1)
    pylab.imshow(images[band])

    pylab.subplot(1, 2, 2)
    pylab.imshow(mimage)

    pylab.show()






