import numpy as np,time
from math import log10
import pymc
import sys
import pyfits
from photofit import imageFit, indexTricks as iT
import pickle

#configfile = 'one_sersic_nomask_mcmc'
configfile = 'one_sersic_nomask_emcee'

def read_config(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    config = {'data_dir':'./', 'output_dir':'./', 'filters':None, 'zeropoints':None, 'filename':None, 'science_tag':'_sci.fits', 'sigma_tag':'_sig.fits', 'psf_tag':'_psf.fits', 'output_tag': '_resid.fits', 'Nsteps':10000, 'logptol':1., 'burnin':None, 'maskname':None, 'fit_type':'MCMC', 'components':None, 'config_file':filename}

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
    config['zeropoints'] = np.array(config['zeropoints'].split(','), dtype='float')
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
    outlines.append('logp %5.3f\n'%results['MLmodel']['logp'])
    
    f = open(results['config']['output_dir']+results['config']['config_file']+'.output','w')
    f.writelines(outlines)
    f.close()
    
    for band in results['config']['filters']:
        ms = []
        for comp in results['MLmodel'][band]:
            ms.append(comp)
        m = np.array(ms).sum(0)
        i = results['IMG'][band].reshape(m.shape)
        s = results['SIGMA'][band].reshape(m.shape)

        hdu = pyfits.PrimaryHDU(i)
        mod = pyfits.ImageHDU(m)
        res = pyfits.ImageHDU(i-m)
        nres = pyfits.ImageHDU((i-m)/s)
        mask = pyfits.ImageHDU(results['MASK'])
        hdulist = pyfits.HDUList([hdu, mod, res, nres, mask])
        hdulist.writeto(results['config']['filename']+'_'+band+'_'+results['config']['config_file']+'.fits', clobber=True)

    if results['config']['fit_type'] == 'MCMC':
        f = open(results['config']['output_dir']+results['config']['config_file']+'.mcmc','w')
        pickle.dump(results['trace'], f)
        f.close()

config = read_config(configfile)
results = imageFit.do_fit(config)
#make_output_files(results)

import pylab
pars = ['q1', 'pa1', 'x_u1', 'y_u1', 're1', 'n1']
for p in pars:
    for i in range(results['trace'][p].shape[0]):
        pylab.plot(results['trace'][p][i])
    pylab.title(p)
    pylab.show()
