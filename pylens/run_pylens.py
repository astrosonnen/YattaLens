import numpy as np,time
from math import log10
import pymc
import indexTricks as iT
import sys
import pyfits
from pylens import pylens, SBModels, MassModels
from photofit import imageFit
import pickle


configfile = sys.argv[1]
config = pylens.read_config(configfile)

image = {}
sigmas = {}
light_models = {}
source_models = {}
lens_models = []
zp = {}
pars = []
guess = []
lower = []
upper = []
steps = []
filters = [filt for filt in config['filters']]

#defines model parameters
par2index = {}
index2par = {}
ncomp = 0
npar = 0
for comp in config['light_components']:
    ncomp += 1
    for par in comp['pars']:
        parpar = comp['pars'][par]
        if parpar['link'] is None and parpar['var'] == 1:
            pars.append(pymc.Uniform(par+str(ncomp), lower=parpar['low'], upper=parpar['up'], value=parpar['value']))
            guess.append(parpar['value'])
            steps.append(parpar['step'])
            par2index['light'+str(ncomp)+'.'+par] = npar
            index2par[npar] = 'light'+str(ncomp)+'.'+par
            npar += 1

ncomp = 0
for comp in config['source_components']:
    ncomp += 1
    for par in comp['pars']:
        parpar = comp['pars'][par]
        if parpar['link'] is None and parpar['var'] == 1:
            pars.append(pymc.Uniform(par+str(ncomp), lower=parpar['low'], upper=parpar['up'], value=parpar['value']))
        guess.append(parpar['value'])
        steps.append(parpar['step'])
        par2index['source'+str(ncomp)+'.'+par] = npar
        index2par[npar] = 'source'+str(ncomp)+'.'+par
        npar += 1

ncomp = 0
for comp in config['lens_components']:
    ncomp += 1
    for par in comp['pars']:
        parpar = comp['pars'][par]
        if parpar['link'] is None and parpar['var'] == 1:
            pars.append(pymc.Uniform(par+str(ncomp), lower=parpar['low'], upper=parpar['up'], value=parpar['value']))
        guess.append(parpar['value'])
        steps.append(parpar['step'])
        par2index['lens'+str(ncomp)+'.'+par] = npar
        index2par[npar] = 'lens'+str(ncomp)+'.'+par
        npar += 1

i = 0

for band in config['filters']:

    zp[band] = config['zeropoints'][i]

    hdu = pyfits.open(config['data_dir']+config['filename']+'_%s'%band+config['science_tag'])[0]

    img = hdu.data.copy()
    subimg = img.copy()
    image[band] = subimg
    subvar = pyfits.open(config['data_dir']+config['filename']+'_%s'%band+config['var_tag'])[0].data.copy()

    sigmas[band] = subvar**0.5

    psf = pyfits.open(config['data_dir']+config['filename']+'_%s'%band+config['psf_tag'])[0].data.copy()
    m = (psf[:2].mean()+psf[-2:].mean()+psf[:,:2].mean()+psf[:,-2:].mean())/4.
    psf -= m
    psf /= psf.sum()

ncomp = 1
for comp in config['lens_components']:

    name = 'lens%d'%ncomp

    pars_here = {}
    for par in comp['pars']:
        if comp['pars'][par]['link'] is None:
            if comp['pars'][par]['var'] == 1:
                pars_here[par] = pars[par2index[name+'.'+par]]
            else:
                pars_here[par] = comp['pars'][par]['value']
        else:
            pars_here[par] = pars[par2index[comp['pars'][par]['link']]]

    ncomp += 1

    lens = MassModels.PowerLaw(name, pars_here)
    lens_models.append(lens)

ncomp = 1
light_pardicts = []
for comp in config['light_components']:

    name = 'light%d'%ncomp

    pars_here = {}
    for par in comp['pars']:
        if comp['pars'][par]['link'] is None:
            if comp['pars'][par]['var'] == 1:
                pars_here[par] = pars[par2index[name+'.'+par]]
            else:
                pars_here[par] = comp['pars'][par]['value']
        else:
            pars_here[par] = pars[par2index[comp['pars'][par]['link']]]

    ncomp += 1

    light_pardicts.append(pars_here)

ncomp = 1
source_pardicts = []
for comp in config['source_components']:

    name = 'source%d'%ncomp

    pars_here = {}
    for par in comp['pars']:
        if comp['pars'][par]['link'] is None:
            if comp['pars'][par]['var'] == 1:
                pars_here[par] = pars[par2index[name+'.'+par]]
            else:
                pars_here[par] = comp['pars'][par]['value']
        else:
            pars_here[par] = pars[par2index[comp['pars'][par]['link']]]

    ncomp += 1

    source_pardicts.append(pars_here)

for band in config['filters']:

    light_models[band] = []
    source_models[band] = []

    for pardict in light_pardicts:

        light = SBModels.Sersic('light', pardict)
        light_models[band].append(light)

    for pardict in source_pardicts:

        source = SBModels.Sersic('source', pardict)
        source_models[band].append(source)

if config['maskname'] is not None:
    MASK = pyfits.open(config['data_dir']+config['maskname'])[0].data.copy()
else:
    MASK = 0.*subimg

mask = MASK==0
mask_r = mask.ravel()


