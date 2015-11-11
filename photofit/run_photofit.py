import numpy,time
from photofit.modelSB import linearmodelSB
from math import log10
import pymc
import indexTricks as iT
from scipy.optimize import basinhopping as bh
import sys
import pyfits
from photofit import models as photmodels, convolve

#configfile = sys.argv[1]

def read_config(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    
    config = {'data_dir':'./', 'output_dir':'./', 'filters':None, 'zeropoints':None, 'filename':None, 'science_tag':'_sci.fits', 'sigma_tag':'_sig.fits', 'psf_tag':'_psf.fits', 'output_tag': '_resid.fits', 'Nsteps':10000, 'maskname':None, 'fit_type':'MCMC', 'components':None, 'config_file':filename}
    
    allowed_models = ['Sersic']
    
    preamble = True
    
    i = 0
    while preamble and i < len(lines):
        if '#' in lines[i] and 'MODELS' in lines[i]:
	    preamble = False
        else:
	    parname = lines[i].split()[0].split(':')[0]
	    if parname in config:
		config[parname] = lines[i].split(':')[1].split('\n')[0].lstrip()
        i += 1
    
    filtlist = []
    filternames = config['filters'].split(',')
    for name in filternames:
        filtlist.append(name.lstrip())
    config['filters'] = filtlist
    config['zeropoints'] = numpy.array(config['zeropoints'].split(','), dtype='float')
    config['Nsteps'] = int(config['Nsteps'])
    
    
    reading_model = False
    components = []
   
    for j in range(i, len(lines)):
        line = lines[j].split()
        if lines[j][0] != '#' and len(line) > 0:
	    if 'model_class' in line[0]:
		model_class = line[1].lstrip()
		if not model_class in allowed_models:
		    print 'model class %s not defined'%model_class
	            df
		if reading_model:
		    components.append(comp)
		reading_model = True
		comp = {'class':model_class, 'pars':{}}

	    else:
		if lines[j][0]!='#' and len(line) > 5:
		    par = line[0]
    		    link = None
    		    if len(line) > 6:
		        link = line[6]
		    tmp_par = {'value': float(line[1]), 'low': float(line[2]), 'up': float(line[3]), \
			       'cov': float(line[4]), 'var': int(line[5]), 'link':link}
		    comp['pars'][par] = tmp_par
        j += 1
    components.append(comp)
    
    config['components'] = components
    
    return config


def do_mcmc(config):

    image = {}
    sigmas = {}
    models = {}
    ZP = {}
    pars = []
    covs = []
    filters = [filt for filt in config['filters']]

    #defines model parameters
    par2index = {}
    ncomp = 0
    npar = 0
    for comp in config['components']:
	ncomp += 1
	for par in comp['pars']:
	    parpar = comp['pars'][par]
	    if parpar['link'] is None and parpar['var'] == 1:
		pars.append(pymc.Uniform(par+str(ncomp), lower=parpar['low'], upper=parpar['up'], \
		value=parpar['value']))
		covs.append(parpar['cov'])
		par2index[str(ncomp)+'.'+par] = npar
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

	"""
	constructs a model for each band.
	for each band, needs to know:
	- How many components is the model made of, and of which kind
	- What parameter corresponds to what
	- What are the initial values of the model parameter, for each component
	- For each model parameter, what is its initial value
	- For each model parameter, whether it is variable or fixed
	- If variable, whether it is linked to another parameter or not.
	"""

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
    M.use_step_method(pymc.AdaptiveMetropolis, pars, cov=numpy.diag(covs))
    M.sample(config['Nsteps'], config['Nsteps']/10)

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

    output['models'] = models
    output['trace'] = trace
    output['MLmodel'] = MLmodel
    output['MLmags'] = MLmags
    output['config'] = config

    return output


def make_output_files(results):
    
    strpars = ['re', 'n', 'pa', 'q']
    outlines = []
    outlines.append('#par mean std\n')
    ncomp = 0
    for comp in results['config']['components']:
	ncomp += 1
	outlines.append('#%s\n'%(comp['class']+'_%d'%ncomp))
	strdone = False
	for band in results['config']['filters']:
	    if not strdone:
		for par in strpars:
		    outlines.append('%s %5.3f\n'%(par, results['models'][band][ncomp-1].values[par]))
		strdone = True
	    for par in ['x', 'y']:
		outlines.append('%s_%s %5.3f\n'%(par, band, results['models'][band][ncomp-1].values[par]))
	for band in results['config']['filters']:
	    magname = comp['class']+'_%s_%d'%(band, ncomp)
	    val = results['MLmags'][magname]
	    outlines.append(band+' %5.3f\n'%val)
    
   
    """
    strpars = ['re', 'n', 'pa', 'q']
    outlines = []
    outlines.append('#par mean std\n')
    ncomp = 0
    for comp in results['config']['components']:
	ncomp += 1
	outlines.append('#%s\n'%(comp['class']+'_%d'%ncomp))
	for par in strpars:
	    if comp['pars'][par]['var'] == 1:
		parname = par+str(ncomp)
		val = results['trace'][parname].mean()
		err = results['trace'][parname].std()
		outlines.append(parname+' %5.3f %5.3f\n'%(val, err))
	for band in results['config']['filters']:
	    for par in ['x', 'y']:
		if comp['pars'][par+'_'+band]['var'] == 1 and comp['pars'][par+'_'+band]['link'] is None:
		    parname = par+'_%s%d'%(band, ncomp)
		    val = results['trace'][parname].mean()
		    err = results['trace'][parname].std()
		    outlines.append(parname+' %5.3f %5.3f\n'%(val, err))
	for band in results['config']['filters']:
	    magname = comp['class']+'_%s_%d'%(band, ncomp)
	    val = results['trace'][magname].mean()
	    err = results['trace'][magname].std()
	    outlines.append(magname+' %5.3f %5.3f\n'%(val, err))
    """
    
    f = open(results['config']['output_dir']+results['config']['config_file']+'.output','w')
    f.writelines(outlines)
    f.close()
    

#config = read_config(configfile)

#if config['fit_type'] == 'MCMC':
#    results = do_mcmc(config)
#elif config['fit_type'] == 'siman':
#    results = do_siman(config)



