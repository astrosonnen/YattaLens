import numpy,time
from spasmoid.modelSB import linearmodelSB
from math import log10

def optimize(data,niter,oname=None,first=True):
    import pymc,pyfits,numpy
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


    @pymc.deterministic(trace=False)
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
        return lp,mags


    @pymc.deterministic
    def lp(lpAM=logpAndMags):
        return lpAM[0]
    
    @pymc.deterministic
    def Mags(lpAM=logpAndMags):
        return lpAM[1]

    @pymc.observed
    def logpCost(value=0.,logP=lp):
        return logP

    costs = [logpCost]
    if priors is not None:
        @pymc.observed
        def colorPrior(value=0.,M=Mags):
            lp = 0.
            for p in priors:
                color = M[model2index[p[0]]]-M[model2index[p[1]]]
                lp += p[2](color)
            return lp
        costs.append(colorPrior)

    def resid(p):
        model = numpy.empty(0)
        for key in filters:
            indx = key2index[key]
            if doSigma==True:
                sigma = sigmas[indx].value
            else:
                sigma = sigmas[key]
            simage = (image[key]/sigma)[mask_r]
            model = numpy.append(model,linearmodelSB(p,simage,sigma[mask_r],mask,models[key],xc,yc,levMar=True,OVRS=OVRS))
        return model


    print "Optimizing",niter
    from SampleOpt import AMAOpt as Opt,levMar as levMar
    default = numpy.empty(0)
    for key in filters:
        indx = key2index[key]
        if doSigma==True:
            sigma = sigmas[indx].value
        else:
            sigma = sigmas[key]
        simage = (image[key]/sigma)[mask_r]
        default = numpy.append(default,simage)
#    levMar(pars,resid,default)

    cov = None
    if 'COV' in data.keys():
        cov = data['COV']

    O = Opt(pars,costs,[lp,Mags],cov=cov)
    O.set_minprop(len(pars)*2)
    O.sample(niter/10)

    O = Opt(pars,costs,[lp,Mags],cov=cov)
    O.set_minprop(len(pars)*2)
    O.cov = O.cov/4.
    O.sample(niter/4)

    O = Opt(pars,costs,[lp,Mags],cov=cov)
    O.set_minprop(len(pars)*2)
    O.cov = O.cov/10.
    O.sample(niter/4)

    O = Opt(pars,costs,[lp,Mags],cov=cov)
    O.set_minprop(len(pars)*2)
    O.cov = O.cov/10.
    O.sample(niter)
    logp,trace,result = O.result()
    mags = numpy.array(result['Mags'])

    for key in model2index.keys():
        result[key] = mags[:,model2index[key]].copy()
    del result['Mags']

    output = {}
    for key in filters:
        indx = key2index[key]
        if doSigma==True:
            sigma = sigmas[indx].value
        else:
            sigma = sigmas[key]
        simage = (image[key]/sigma)[mask_r]
        m = linearmodelSB([p.value for p in pars],simage,sigma[mask_r],mask,models[key],xc,yc,noResid=True,OVRS=OVRS)
        output[key] = m
    return output,(logp,trace,result)
