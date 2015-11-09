import numpy as np
import pymc

#fits parametrized surface brigthness profiles to photometry data in multiple bands.

def do_fit(science, sigma, models, mask, filters, zp, pars, cov=None):

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

