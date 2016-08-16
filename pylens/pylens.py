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


def getModel(lenses, light_profiles, source_profiles, image, sigma, X, Y, zp=30., shear=None, returnImg=False, mask=None):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = image.flatten()
    S = sigma.flatten()

    if type(lenses) != type([]):
        lenses = [lenses]
    if type(light_profiles) != type([]):
        light_profiles = [light_profiles]
    if type(source_profiles) != type([]):
        source_profiles = [source_profiles]

    if shear is not None:
        shear.setPars()
        lenses += [shear]

    modlist = []

    if mask is None:
        mask = np.ones(image.size, dtype=bool)

# We need to update the parameters of the objects to the values proposed by the optimizer
    for lens in lenses:
        lens.setPars()

    xl, yl = getDeflections(lenses, [X, Y])

    for light in light_profiles:
        light.setPars()
        light.amp = 1.
        lmodel = (convolve.convolve(light.pixeval(X, Y), light.convolve, False)[0].ravel()/S)
        modlist.append(lmodel[mask])

    for source in source_profiles:
        source.setPars()
        source.amp = 1.
        smodel = (convolve.convolve(source.pixeval(xl, yl), source.convolve, False)[0].ravel()/S)
        modlist.append(smodel[mask])

    model = np.array(modlist).T

    if np.isnan(model).any():
        return -np.inf, (99., 99.)

    amps, chi = optimize.nnls(model, (I/S)[mask])

    nlight = len(light_profiles)
    nsource = len(source_profiles)

    lmags = []
    for i in range(nlight):
        light_profiles[i].amp *= amps[i]
        if amps[i] <=0.:
            lmags.append(99.)
        else:
            lmags.append(light_profiles[i].Mag(zp))

    for i in range(nsource):
        source_profiles[i].amp *= amps[i+nlight]
        if amps[i+nlight] <=0.:
            lmags.append(99.)
        else:
            lmags.append(source_profiles[i].Mag(zp))

    logp = -0.5*chi

    if returnImg:
        mimages = []
        for light in light_profiles:
            lmodel = (convolve.convolve(light.pixeval(X, Y), light.convolve, False)[0].ravel())
            mimages.append(lmodel.reshape(image.shape))
        for source in source_profiles:
            lmodel = (convolve.convolve(source.pixeval(xl, yl), source.convolve, False)[0].ravel())
            mimages.append(lmodel.reshape(image.shape))

        return logp, lmags, mimages

    else:
        return logp, lmags


def getModel_fixedamps(lenses, light_profiles, source_profiles, amps, image, sigma, X, Y, zp=30., shear=None, \
                       returnImg=False, mask=None):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = image.flatten()
    S = sigma.flatten()

    if type(lenses) != type([]):
        lenses = [lenses]
    if type(light_profiles) != type([]):
        light_profiles = [light_profiles]
    if type(source_profiles) != type([]):
        source_profiles = [source_profiles]

    if shear is not None:
        shear.setPars()
        lenses += [shear]

    modlist = []

    if mask is None:
        mask = np.ones(image.size, dtype=bool)

# We need to update the parameters of the objects to the values proposed by the optimizer
    for lens in lenses:
        lens.setPars()

    xl, yl = getDeflections(lenses, [X, Y])

    nlight = len(light_profiles)
    nsource = len(source_profiles)

    lmodel = 0.*I
    for i in range(nlight):
        light = light_profiles[i]
        light.setPars()
        light.amp = amps[i]
        lmodel += (convolve.convolve(light.pixeval(X, Y), light.convolve, False)[0].ravel()/S)

    modlist.append(lmodel[mask])

    for source in source_profiles:
        source.setPars()
        source.amp = 1.
        smodel = (convolve.convolve(source.pixeval(xl, yl), source.convolve, False)[0].ravel()/S)
        modlist.append(smodel[mask])

    model = np.array(modlist).T

    if np.isnan(model).any():
        return -np.inf, (99., 99.)

    amps, chi = optimize.nnls(model, (I/S)[mask])

    lmags = []
    for i in range(nlight):
        light_profiles[i].amp *= amps[0]
        if amps[0] <=0.:
            lmags.append(99.)
        else:
            lmags.append(light_profiles[i].Mag(zp))
    light_profiles[nlight-1].amp *= amps[1]

    for i in range(nsource):
        source_profiles[i].amp *= amps[i+1]
        if amps[i+1] <=0.:
            lmags.append(99.)
        else:
            lmags.append(source_profiles[i].Mag(zp))

    logp = -0.5*chi

    if returnImg:
        mimages = []
        for light in light_profiles:
            lmodel = (convolve.convolve(light.pixeval(X, Y), light.convolve, False)[0].ravel())
            mimages.append(lmodel.reshape(image.shape))
        for source in source_profiles:
            lmodel = (convolve.convolve(source.pixeval(xl, yl), source.convolve, False)[0].ravel())
            mimages.append(lmodel.reshape(image.shape))

        return logp, lmags, mimages

    else:
        return logp, lmags



def do_fit(pars, cov, bands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, do_convol=True, Nsamp=11000, \
           burnin=1000):

    @pymc.deterministic(trace=False)
    def logpAndMags(p=pars):
        sumlogp = 0.
        magslist = []
        i = 0
        for band in bands:
            logp, mags = getModel(lens, lights[band], sources[band], images[band], sigmas[band], \
                                         X, Y, zp=zps[band], mask=mask_r, do_convol=do_convol)
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



def do_fit_emcee(pars, bands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, shear=None, nwalkers=50, nsamp=100,\
                 gaussprior=None, stepsize=None, do_convol=True):

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
                                         X, Y, zp=zps[band], mask=mask_r, shear=shear, do_convol=do_convol)
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
            if gaussprior is not None:
                if gaussprior[j]:
                    a, b = (bounds[j][0] - pars[j].value)/stepsize[j], (bounds[j][1] - pars[j].value)/stepsize[j]
                    #p0 = np.random.normal(pars[j].value, steps[j], 1)
                    p0 = truncnorm.rvs(a, b, size=1)*stepsize[j] + pars[j].value
            tmp[j] = p0

        start.append(tmp)


    print "Sampling"

    sampler.run_mcmc(start, nsamp)

    output = {'chain': sampler.chain, 'logp': sampler.lnprobability}

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    return output

def do_fit_emcee_inputwalkers(candidate, pars, fitbands, start, mask_r, nsamp=500):

    bounds = []
    for par in pars:
        bounds.append((par.parents['lower'], par.parents['upper']))

    npars = len(pars)

    def logprior(allpars):
        for i in range(0, npars):
            if allpars[i] < bounds[i][0] or allpars[i] > bounds[i][1]:
                return -np.inf
        return 0.

    nwalkers = len(start)

    def logpfunc(allpars):
        lp = logprior(allpars)
        if not np.isfinite(lp):
            return -np.inf

        for j in range(0, npars):
            pars[j].value = allpars[j]
        sumlogp = 0.
        i = 0

        for band in fitbands:
            logp, mags = getModel(candidate.lens_model, candidate.light_model[band], candidate.source_model[band], \
                                  candidate.sci[band], candidate.err[band], candidate.X, candidate.Y, \
                                  zp=candidate.zp[band], mask=mask_r)

            if logp != logp:
                return -np.inf
            sumlogp += logp
            i += 1

        return sumlogp

    sampler = emcee.EnsembleSampler(nwalkers, npars, logpfunc)

    print "Sampling"

    sampler.run_mcmc(start, nsamp)

    output = {'chain': sampler.chain, 'logp': sampler.lnprobability}

    ML = sampler.flatlnprobability.argmax()

    for j in range(0, npars):
        pars[j].value = sampler.flatchain[ML, j]

    return output



