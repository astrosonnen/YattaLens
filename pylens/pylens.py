import numpy as np
from scipy import optimize
from scipy.optimize import fmin_slsqp
from photofit import convolve
import pymc, emcee
from scipy.stats import truncnorm

class LensModel:
    """
    IN PROGRESS
    The purpose of this class is to allow lens modelling with e.g., caching to
        update parts of the model independently.
    """
    def __init__(self,img,sig,gals,lenses,srcs,psf=None):
        self.img = img
        self.sig = sig
        self.gals = gals
        self.lenses = lenses
        self.srcs = srcs
        self.psf = psf

        self.rhs = (img/sig).flatten()
        self.fsig = sig.flatten()
        self.fit

        self.mask = None

        self.logp = None
        self.model = None
        self.amps = None
        self.components = None

    def addMask(self,mask):
        self.rhs = (img/sig)[mask].flatten()
        self.fsig = sig[mask].flatten()

    def fit(x,y,fast=False):
        mask = self.mask
        gals = self.gals
        srcs = self.srcs
        lenses = self.lenses
        rhs = self.rhs
        sig = self.fsig

        if mask is not None:
            xin = x[mask].copy()
            yin = y[mask].copy()
        else:
            xin = x.copy()
            yin = y.copy()

        model = numpy.empty((len(gals)+len(srcs),rhs.size))
        n = 0
        for gal in gals:
            gal.setPars()
            gal.amp = 1.
            tmp = gal.pixeval(xin,yin,1./self.oversample,csub=self.csub)
            if numpy.isnan(tmp):
                self.model = None
                self.amps = None
                self.components = None
                self.logp = -1e300
                return -1e300
            if psf is not None and gal.convolve is not None:
                if mask is None:
                    model[n] = convolve.convolve(tmp,psf,False)[0]
                else:
                    model[n] = 1 
    
        gals = self.gals


def getDeflections(massmodels,points):
    if type(points)==type([]) or type(points)==type(()):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    if type(massmodels)!=type([]):
        massmodels = [massmodels]
    x0 = x.copy()
    y0 = y.copy()
    for massmodel in massmodels:
        xmap,ymap = massmodel.deflections(x,y)
        y0 -= ymap
        x0 -= xmap
    return x0.reshape(x.shape),y0.reshape(y.shape)



def lens_images(massmodels,sources,points,factor=1,getPix=False):
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    if type(massmodels)!=type([]):
        massmodels = [massmodels]
#    x0 = x.flatten()
#    y0 = y.flatten()
    x0 = x.copy()
    y0 = y.copy()
    for massmodel in massmodels:
        xmap,ymap = massmodel.deflections(x,y)
        y0 -= ymap#.reshape(y.shape)#/scale
        x0 -= xmap#.reshape(x.shape)#/scale
    x0,y0 = x0.reshape(x.shape),y0.reshape(y.shape)
    if getPix==True:
        return x0,y0
    if type(sources)!=type([]):
        sources = [sources]
    out = x*0.
    for src in sources:
        out += src.pixeval(x0,y0,factor,csub=11)
    return out


def dblPlane(scales,massmodels,sources,points,factor):
    if type(points)==type([]):
        x1,y1 = points[0].copy(),points[1].copy()
    else:
        y1,x1 = points[0].copy(),points[1].copy()
    out = x1*0.
    ax_1 = x1*0.
    ay_1 = x1*0.
    for l in massmodels[0]:
        xmap,ymap = l.deflections(x1,y1)
        ax_1 += xmap.reshape(ax_1.shape)
        ay_1 += ymap.reshape(ay_1.shape)
    for s in sources[0]:
        out += s.pixeval(x1-ax_1,y1-ay_1,factor,csub=11)
    x2 = x1-scales[0,0]*ax_1
    y2 = y1-scales[0,0]*ay_1
    ax_2 = x2*0.
    ay_2 = y2*0.
    for l in massmodels[1]:
        xmap,ymap = l.deflections(x2,y2)
        ax_2 += xmap.reshape(ax_2.shape)
        ay_2 += ymap.reshape(ay_2.shape)
    for s in sources[1]:
        out += s.pixeval(x1-ax_1-ax_2,y1-ay_1-ay_2,factor,csub=11)
    return out


def multiplePlanes(scales,massmodels,points):
    from numpy import zeros,eye,triu_indices
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    out = x*0.
    nplanes = len(massmodels)
    tmp = scales.copy()
    scales = eye(nplanes)
    scales[triu_indices(nplanes,1)] = tmp
    ax = zeros((x.shape[0],x.shape[1],nplanes))
    ay = ax.copy()
    x0 = x.copy()
    y0 = y.copy()
    out = []
    for p in range(nplanes):
        for massmodel in massmodels[p]:
            xmap,ymap = massmodel.deflections(x0,y0)
            ax[:,:,p] += xmap.reshape(x.shape)
            ay[:,:,p] += ymap.reshape(y.shape)
        x0 = x-(ax[:,:,:p+1]*scales[:p+1,p]).sum(2)
        y0 = y-(ay[:,:,:p+1]*scales[:p+1,p]).sum(2)
        out.append([x0,y0])
    return out


def objf(x, lhs, rhs):
    return ((np.dot(lhs, x) - rhs)**3).sum()
def objdf(x, lhs, rhs):
    return np.dot(lhs.T, np.dot(lhs, x) - rhs)

def getModel(lens, light, source, image, sigma, X, Y, zp=30., returnImg=False, mask=None):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = image.flatten()
    S = sigma.flatten()
# We need to update the parameters of the objects to the values proposed by the optimizer
    light.setPars()
    source.setPars()
    lens.setPars()

    light.amp = 1.
    source.amp = 1.

    xl,yl = getDeflections(lens,[X,Y])

    if mask is None:
        mask = np.ones(image.size, dtype=bool)

    lmodel = (convolve.convolve(light.pixeval(X,Y), light.convolve,False)[0].ravel()/S)
    smodel = (convolve.convolve(source.pixeval(xl,yl), source.convolve,False)[0].ravel()/S)
    model = np.array((lmodel[mask], smodel[mask])).T

    if np.isnan(model).any():
        return -np.inf, (99., 99.)

    amps,chi = optimize.nnls(model,(I/S)[mask])

    light.amp *= amps[0]
    source.amp *= amps[1]
    if amps[0] <=0.:
        lmag = 99.
    else:
        lmag = light.Mag(zp)
    if amps[1] <= 0.:
        smag = 99.
    else:
        smag = source.Mag(zp)

    logp = -0.5*chi - np.log(S[mask].sum())

    if returnImg:
        #fullmodel = np.array((lmodel, smodel))
        #mimage = (amps*(fullmodel*S).T).sum(axis=1)
        limage = amps[0]*lmodel*S
        simage = amps[1]*smodel*S
        return logp, (lmag, smag), (limage.reshape(image.shape), simage.reshape(image.shape))

    else:
        return logp, (lmag, smag)


def getModel_sourceonly(lens, source, image, sigma, X, Y, zp=30., returnImg=False, mask=None):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = image.flatten()
    S = sigma.flatten()
# We need to update the parameters of the objects to the values proposed by the optimizer
    source.setPars()
    lens.setPars()

    source.amp = 1.

    xl,yl = getDeflections(lens,[X,Y])

    if mask is None:
        mask = np.ones(image.size, dtype=bool)

    smodel = (convolve.convolve(source.pixeval(xl,yl), source.convolve,False)[0].ravel()/S)
    model = np.array((smodel[mask], )).T

    if np.isnan(model).any():
        return -np.inf, (99., 99.)

    amps,chi = optimize.nnls(model,(I/S)[mask])

    source.amp *= amps[0]
    if amps[0] <= 0.:
        smag = 99.
    else:
        smag = source.Mag(zp)

    logp = -0.5*chi - np.log(S[mask].sum())

    if returnImg:
        simage = amps[0]*smodel*S
        return logp, smag, simage.reshape(image.shape)

    else:
        return logp, smag


def getModel_lightonly(light, image, sigma, X, Y, zp=30., returnImg=False, mask=None):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = image.flatten()
    S = sigma.flatten()
# We need to update the parameters of the objects to the values proposed by the optimizer
    light.setPars()

    light.amp = 1.

    if mask is None:
        mask = np.ones(image.size, dtype=bool)

    lmodel = (convolve.convolve(light.pixeval(X,Y), light.convolve,False)[0].ravel()/S)
    model = np.array((lmodel[mask], )).T

    if np.isnan(model).any():
        return -np.inf, (99., 99.)

    amps,chi = optimize.nnls(model,(I/S)[mask])

    light.amp *= amps[0]
    if amps[0] <=0.:
        lmag = 99.
    else:
        lmag = light.Mag(zp)

    logp = -0.5*chi

    if returnImg:
        limage = amps[0]*lmodel*S
        return logp, lmag, limage.reshape(image.shape)

    else:
        return logp, lmag



def do_fit(pars, cov, bands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, Nsamp=11000, burnin=1000):

    @pymc.deterministic(trace=False)
    def logpAndMags(p=pars):
        sumlogp = 0.
        magslist = []
        i = 0
        for band in bands:
            logp, mags = getModel(lens, lights[band], sources[band], images[band], sigmas[band], \
                                         X, Y, zp=zps[band], mask=mask_r)
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

    for par in pars:
        trace[str(par)] = M.trace(par)[:]

    trace['logp'] = M.trace('lp')[:]
    magss = np.array(M.trace('Mags')[:])
    i = 0
    for band in bands:
        trace['lens_%s'%band] = magss[:, i, 0]
        trace['source_%s'%band] = magss[:, i, 1]
        i += 1

    ML = trace['logp'].argmax()
    for par in pars:
        mlval = trace[str(par)][ML]
        par.value = mlval

    return trace


def do_fit_lightonly(lpars, lcov, bands, lights, images, sigmas, X, Y, mask_r, zps, Nsamp=11000, burnin=1000):

    @pymc.deterministic
    def lightonlylogp(lp=lpars):
        sumlogp = 0.
        for lband in bands:
            logp, mag = getModel_lightonly(lights[lband], images[lband], sigmas[lband], X, Y, zp=zps[lband], mask=mask_r)

            if logp != logp:
                return -1e300
            sumlogp += logp
        return sumlogp

    @pymc.stochastic(observed=True, name='logp')
    def logp(value=0., lp=lpars):
        return lightonlylogp

    print 'sampling light parameters only'

    M = pymc.MCMC(lpars + [lightonlylogp])
    M.use_step_method(pymc.AdaptiveMetropolis, lpars, cov=np.diag(lcov))
    M.sample(Nsamp, burnin)

    trace = {}

    for par in lpars:
        trace[str(par)] = M.trace(par)[:]

    trace['logp'] = M.trace('lightonlylogp')[:]

    ML = trace['logp'].argmax()
    for par in lpars:
        mlval = trace[str(par)][ML]
        par.value = mlval

    return trace


def do_fit_emcee(pars, bands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, nwalkers=50, nsamp=100,\
                 usecov=None, steps=None):

    bounds = []
    for par in pars:
        bounds.append((par.parents['lower'], par.parents['upper']))

    npars = len(pars)

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

        for band in bands:
            logp, mags = getModel(lens, lights[band], sources[band], images[band], sigmas[band], \
                                         X, Y, zp=zps[band], mask=mask_r)
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
            if usecov is not None:
                if usecov[j]:
                    a, b = (bounds[j][0] - pars[j].value)/steps[j], (bounds[j][1] - pars[j].value)/steps[j]
                    #p0 = np.random.normal(pars[j].value, steps[j], 1)
                    p0 = truncnorm.rvs(a, b, size=1)*steps[j] + pars[j].value
            tmp[j] = p0

        start.append(tmp)


    print "Sampling"

    sampler.run_mcmc(start, nsamp)

    output = {'chain': sampler.chain, 'logp': sampler.lnprobability}

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    return output
