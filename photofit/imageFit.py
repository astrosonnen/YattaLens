import numpy as np,time
from photofit.modelSB import linearmodelSB
from math import log10
import pymc
import indexTricks as iT
import pyfits
from scipy.optimize import basinhopping
from photofit import models as photmodels, convolve


def do_fit(config):

    image = {}
    sigmas = {}
    models = {}
    ZP = {}
    pars = []
    guess = []
    lower = []
    upper = []
    covs = []
    filters = [filt for filt in config['filters']]

    #defines model parameters
    par2index = {}
    index2par = {}
    ncomp = 0
    npar = 0
    for comp in config['components']:
	ncomp += 1
	for par in comp['pars']:
	    parpar = comp['pars'][par]
	    if parpar['link'] is None and parpar['var'] == 1:
		pars.append(pymc.Uniform(par+str(ncomp), lower=parpar['low'], upper=parpar['up'], \
		value=parpar['value']))
		guess.append(parpar['value'])
		lower.append(parpar['low'])
		upper.append(parpar['up'])
		covs.append(parpar['cov'])
		par2index[str(ncomp)+'.'+par] = npar
		index2par[npar] = str(ncomp)+'.'+par
		npar += 1

    i = 0
    for band in config['filters']:
	ZP[band] = config['zeropoints'][i]
	hdu = pyfits.open(config['filename']+'_%s'%band+config['science_tag'])[0]
        img = hdu.data.copy()
        subimg = img.copy()
        image[band] = subimg
        subsigma = pyfits.open(config['filename']+'_%s'%band+config['sigma_tag'])[0].data.copy()
    
        sigmas[band] = subsigma
    
        psf = pyfits.open(config['filename']+'_%s'%band+config['psf_tag'])[0].data.copy()
        m = (psf[:2].mean()+psf[-2:].mean()+psf[:,:2].mean()+psf[:,-2:].mean())/4.
        psf -= m
        psf /= psf.sum()

	parnames = ['x', 'y', 'pa', 'q', 're', 'n']
	this_band_pars = ['x_'+band, 'y_'+band, 'pa', 'q', 're', 'n']

	models[band] = []

	ncomp = 0
	for comp in config['components']:
	    ncomp += 1
	    var = {}
	    const = {'amp':1.}
	    npar = 0
	    for par in this_band_pars:
		if comp['pars'][par]['link'] is None:
		    if comp['pars'][par]['var'] == 1:
			var[parnames[npar]] = par2index[str(ncomp)+'.'+par]
		    elif comp['pars'][par]['var'] == 0:
			const[parnames[npar]] = comp['pars'][par]['value']
		    else:
			df
		else:
		    link = comp['pars'][par]['link'].split('.')
		    lcomp = int(link[0])
		    lpar = link[1]
		    linkedpar = config['components'][lcomp-1]['pars'][lpar]
		    if linkedpar['var'] == 1:
			var[parnames[npar]] = par2index[comp['pars'][par]['link']]
		    elif linkedpar['var'] == 0:
			const[parnames[npar]] = linkedpar['value']
		npar += 1

	    model = photmodels.Sersic('Sersic_%s_%s'%(band, ncomp), var, const)
	    model.convolve = convolve.convolve(subimg, psf)[1]
	    models[band].append(model)

	i += 1

    if config['maskname'] is not None:
	MASK = pyfits.open(config['maskname'])[0].data.copy()
    else:
	MASK = 0.*subimg

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
    OVRS = 1
    yc,xc = iT.overSample(imshape,OVRS)

    for key in filters:
	sigmas[key] = sigmas[key].ravel()

    for key in filters:
        image[key] = image[key].ravel()


    if config['fit_type'] == 'MCMC':

        @pymc.deterministic(trace=False)
        def logpAndMags(p=pars):
            lp = 0.
            mags = []
            for key in filters:
                indx = key2index[key]
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
    
        print "Sampling",config['Nsteps']
    
        M = pymc.MCMC(pars+[lp, logpAndMags, Mags])
        M.use_step_method(pymc.AdaptiveMetropolis, pars, cov=np.diag(covs))
	if config['burnin'] = None:
	    burnin = config['Nsteps']/10
	else:
	    burnin = config['burnin']

        M.sample(config['Nsteps'] + burnin, burnin)
    
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
            trace[key] = M.trace('Mags')[:, model2index[key]].copy()
            MLmags[key] = M.trace('Mags')[ML, model2index[key]].copy()
    
        output = {}
        for key in filters:
            indx = key2index[key]
            sigma = sigmas[key]
            simage = (image[key]/sigma)[mask_r]
            m = linearmodelSB([p.value for p in pars],simage,sigma[mask_r],mask,models[key],xc,yc,noResid=True,OVRS=OVRS)
            MLmodel[key] = m
    
        output['IMG'] = image
        output['SIGMA'] = sigmas
        output['MASK'] = MASK
        output['models'] = models
        output['trace'] = trace
        output['MLmodel'] = MLmodel
        output['MLmags'] = MLmags
        output['config'] = config
    
        return output

    elif config['fit_type'] == 'basinhop':

	bounds = [(low, high) for low, high in zip(lower, upper)]
	minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds, tol=1.)

        def nlogp(p):
            lp = 0.
            for key in filters:
                indx = key2index[key]
                sigma = sigmas[key]
                simage = (image[key]/sigma)[mask_r]
                lp += linearmodelSB(p,simage,sigma[mask_r],mask,models[key],xc,yc,OVRS=OVRS)
	    if lp != lp:
		lp = -1e300
            return -lp
 
	res = basinhopping(nlogp, guess, niter=config['Nsteps'], minimizer_kwargs=minimizer_kwargs)

	MLpars = res.x

	MLmodel = {}
	for i in range(0, len(MLpars)):
	    MLmodel[index2par[i]] = MLpars[i]

	MLmags = {}

        output = {}
        for key in filters:
            indx = key2index[key]
            sigma = sigmas[key]
            simage = (image[key]/sigma)[mask_r]
            m = linearmodelSB(MLpars,simage,sigma[mask_r],mask,models[key],xc,yc,noResid=True,OVRS=OVRS)
            MLmodel[key] = m
	    for i in range(0, len(models[key])):
		MLmags[models[key][i].name] = models[key][i].Mag(ZP[key])
 
        output['IMG'] = image
        output['SIGMA'] = sigmas
        output['MASK'] = MASK
        output['models'] = models
        output['MLmodel'] = MLmodel
        output['config'] = config
	output['MLmags'] = MLmags

	return output
	
    else:
	raise ValueError("fit_type must be one between 'MCMC' and 'basinhop'.")
	
