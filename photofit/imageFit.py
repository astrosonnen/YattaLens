import numpy,time
from photofit.modelSB import linearmodelSB
from math import log10
import pymc

def do_mcmc(data,niter):
    import indexTricks as iT

    priors = data['PRIORS']
    models = data['MODELS']
    pars = data['PARAMS']

    image = {}
    for key in data['IMG'].keys():
        image[key] = data['IMG'][key].copy()
    ZP = data['ZP']
    filters = [filt for filt in data['FILTERS']]

    sigmas = data['SIGMA']
    if 'GAIN' in data.keys():
        gain = data['GAIN']
        doSigma = True
    else:
        doSigma = False

    if 'OVRS' in data.keys():
        OVRS = data['OVRS']
    else:
        OVRS = 1

    MASK = data['MASK'].copy()
    mask = MASK==0
    mask_r = mask.ravel()

    key2index = {}
    i = 0
    for key in filters:
        key2index[key] = i
        i += 1

    model2index = {}
    i = 0
    for key in filters:
        for model in models[key]:
            model2index[model.name] = i
            i += 1

    imshape = MASK.shape
    yc,xc = iT.overSample(imshape,OVRS)

    if doSigma==True:
        nu = {}
        eta = {}
        background = {}
        counts = {}
        sigmask = {}
        for key in filters:
            nu[key] = pymc.Uniform('nu_%s'%key,-6,6,value=log10(gain[key]))
            eta[key] = pymc.Uniform('eta_%s'%key,-4,5,value=1.)
            background[key] = sigmas[key]
            sigmask[key] = image[key]>1.5*sigmas[key]**0.5
            counts[key] = image[key][sigmask[key]].copy()
            pars.append(nu[key])
            pars.append(eta[key])

        def getSigma(n=nu,e=eta,b=background,c=counts,m=mask):
            sigma = b.copy()
            sigma[m] += ((10**n)*c)**e
            return numpy.sqrt(sigma).ravel()

        sigmas = []
        for key in filters:
            parents = {'n':nu[key],'e':eta[key],'b':background[key],
                    'c':counts[key],'m':sigmask[key]}
            sigmas.append(pymc.Deterministic(eval=getSigma,
                        name='sigma_%s'%key,parents=parents,doc='',
                        trace=False,verbose=False))
    else:
        for key in filters:
            sigmas[key] = sigmas[key].ravel()

    for key in filters:
        image[key] = image[key].ravel()

    @pymc.deterministic()#(trace=False)
    def logpAndMags(p=pars):
        lp = 0.
        mags = []
        for key in filters:
            indx = key2index[key]
            if doSigma==True:
                sigma = sigmas[indx].value
            else:
                sigma = sigmas[key]
            simage = (image[key]/sigma)[mask_r]
            lp += linearmodelSB(p,simage,sigma[mask_r],mask,models[key],xc,yc,OVRS=OVRS)
            mags += [model.Mag(ZP[key]) for model in models[key]]
	if lp != lp:
	    lp = -1e300
        return lp,mags


    @pymc.deterministic
    def lp(lpAM=logpAndMags):
        return lpAM[0]
    
    @pymc.deterministic(name='Mags')
    def Mags(lpAM=logpAndMags):
        return lpAM[1]

    @pymc.stochastic(observed=True, name='logp')
    def logpCost(value=0., p=pars):
        return lp

    cov = None
    if 'COV' in data.keys():
        cov = data['COV']

    print "Sampling",niter

    M = pymc.MCMC(pars+[lp, logpAndMags, Mags])
    M.use_step_method(pymc.AdaptiveMetropolis, pars, cov=numpy.diag(cov))
    M.sample(niter, niter/10)

    trace = {}
    for par in pars:
        trace[str(par)] = M.trace(par)[:]
    trace['logp'] = M.trace('lp')[:]

    ML = trace['logp'].argmax()

    MLmodel = {}
    for par in pars:
        MLmodel[str(par)] = trace[str(par)][ML]

    MLmags = {}
    for key in model2index.keys():
        #MLmags[key] = mags[:,model2index[key]].copy()
        trace[key] = M.trace('Mags')[:, model2index[key]].copy()
        MLmags[key] = M.trace('Mags')[ML, model2index[key]].copy()

    output = {}
    for key in filters:
        indx = key2index[key]
        if doSigma==True:
            sigma = sigmas[indx].value
        else:
            sigma = sigmas[key]
        simage = (image[key]/sigma)[mask_r]
        m = linearmodelSB([p.value for p in pars],simage,sigma[mask_r],mask,models[key],xc,yc,noResid=True,OVRS=OVRS)
        MLmodel[key] = m
    return trace, MLmodel, MLmags
