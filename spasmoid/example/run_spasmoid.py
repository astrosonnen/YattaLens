from spasmoid import imageFit as modelSB, models, convolve
import os,sys,numpy,pymc,pyfits
from math import pi
from scipy import ndimage
import warnings
import sys

configfile = sys.argv[1]

f = open(configfile, 'r')
lines = f.readlines()
f.close()

input_pars = {'data_dir':'./', 'output_dir':'./', 'filter_names':None, 'zeropoints':None, 'filename':None, 'science_tag':'_sci.fits', 'sigma_tag':'_sig.fits', 'psf_tag':'_psf.fits', 'output_tag': '_resid.fits', 'Nsteps':10000}

allowed_models = ['Sersic']

preamble = True

i = 0
while preamble and i < len(lines):
    if '#' in lines[i] and 'MODELS' in lines[i]:
	preamble = False
    else:
	parname = lines[i].split()[0].split(':')[0]
	if parname in input_pars:
	    input_pars[parname] = lines[i].split(':')[1].split('\n')[0].lstrip()
    i += 1

filtlist = []
filternames = input_pars['filter_names'].split(',')
for name in filternames:
    filtlist.append(name.lstrip())
input_pars['filter_names'] = filtlist
input_pars['zeropoints'] = numpy.array(input_pars['zeropoints'].split(','), dtype='float')
input_pars['Nsteps'] = int(input_pars['Nsteps'])


reading_model = False
components = []
ncomp = 0
counter = 0
var_counter = 0
const_counter = 0

mcmcpars = []
covs = []

allpars = {}
var_indices = {}
const_indices = {}
all_indices = {}

for j in range(i, len(lines)):
    line = lines[j].split()
    if lines[j][0] != '#' and len(line) > 0:
	if 'model_class' in line[0]:
	    model_class = line[1].lstrip()
	    components.append(model_class)
	    if not model_class in allowed_models:
		print 'model class %s not defined'%model_class
		df
	    ncomp += 1

	else:
	    if lines[j][0]!='#' and len(line) > 5:
		par = line[0] + str(ncomp)
		guess = float(line[1])
		low = float(line[2])
		up = float(line[3])
		cov = float(line[4])
		var = int(line[5])
		all_indices[counter] = par
		link = None
		if len(line) > 6:
		    link = int(line[6])
		allpars[par] = {'value': guess, 'var': var, 'index':counter, 'link':link}
		if link is None:
		    if var==1:
			var_indices[par] = var_counter
			mcmcpars.append(pymc.Uniform(par, low, up, value=guess))
			covs.append(cov)
			var_counter += 1
		    else:
			const_indices[par] = const_counter
			const_counter += 1
		counter += 1

    j += 1

#EXAMPLE: fits two Sersic components.

galaxyname = 'SL2SJ021411-040502'

maskname = galaxyname+'_mask.fits'

niter = 10000 #number of steps in the optimization chain

#
#
#
zeropoints = {}
exptimes = {}
filters = []
nfilt = len(input_pars['filter_names'])
for i in range(0, nfilt):
    band = input_pars['filter_names'][i]
    filters.append(band)
    zeropoints[band] = input_pars['zeropoints'][i]
    exptimes[band] = 1.

#
#  Package the input for fitting
#
data = {'MODELS':{},'IMG':{},'SIGMA':{},'ZP':zeropoints,'FILTERS':filters}
data['PRIORS'] = None

gain = {}
back = {}
n = len(mcmcpars)
i = 0
for band in filters:
    hdu = pyfits.open(input_pars['filename']+'_%s'%band+input_pars['science_tag'])[0]
    img = hdu.data.copy()
    subimg = img.copy()
    data['IMG'][band] = subimg
    subsigma = pyfits.open(input_pars['filename']+'_%s'%band+input_pars['sigma_tag'])[0].data.copy()

    data['SIGMA'][band] = subsigma

    psf = pyfits.open(input_pars['filename']+'_%s'%band+input_pars['psf_tag'])[0].data.copy()
    m = (psf[:2].mean()+psf[-2:].mean()+psf[:,:2].mean()+psf[:,-2:].mean())/4.
    psf -= m
    psf /= psf.sum()

    this_band_pars = ['x_'+band, 'y_'+band, 'pa', 'q', 're', 'n']

    data['MODELS'][band] = []

    for j in range(1, ncomp+1):
	var = {}
	const = {'amp':1.}
	for par in this_band_pars:
	    parname = par
	    if 'x_' in parname:
		parname = 'x'
	    elif 'y_' in parname:
		parname = 'y'
	    
	    if allpars[par+str(j)]['link'] is None:
		if allpars[par+str(j)]['var'] == 1:
		    var[parname] = var_indices[par+str(j)]
		else:
		    const[parname] = const_indices[par+str(j)]
	    else:
		linked_par = all_indices[allpars[par+str(j)]['link']]
		if allpars[linked_par]['var'] == 1:
		    var[parname] = var_indices[linked_par]
		else:
		    const[parname] = const_indices[linked_par]

	model = models.Sersic('sersic_%s_%s'%(filters[i], j), var, const)
	model.convolve = convolve.convolve(subimg,psf)[1]
	data['MODELS'][band].append(model)

    i += 1

if os.path.isfile(maskname):
    mask = pyfits.open(maskname)[0].data.copy()
    mask[mask>0] = 1 #this step is necessary if using segmentation maps generated by sextractor
else:
    warnings.warn("Warning: mask file not found")
    mask = numpy.zeros(subimg.shape)


data['MASK'] = mask.copy()

data['PARAMS'] = mcmcpars
data['COV'] = numpy.asarray(covs)
data['OVRS'] = 1

vals = modelSB.optimize(data,1000)#input_pars['Nsteps'])
logp,trace,dets = vals[1]
print logp[0],logp[-1]


output = []
j=0
for par in mcmcpars:
    val = trace[-1][j]
    j = j+1
    print par,val
    output.append(str(par)+' %5.3f\n'%val)

for f in filters:
    for j in range(1, ncomp+1):
	mag = dets['sersic_%s_%s'%(f, j)][-1]
	print f,mag
	output.append(f+' %4.2f\n'%mag)

f = open('output.txt','w')
f.writelines(output)
f.close()


for filt in filters:
    i = data['IMG'][filt]
    s = data['SIGMA'][filt]
    m = numpy.array(vals[0][filt])

    m = m.sum(0)
    hdu = pyfits.PrimaryHDU(i-m)
    nres= pyfits.ImageHDU((i-m)/s.reshape(i.shape))
    sigh= pyfits.ImageHDU(m.reshape(i.shape))
    hdulist = pyfits.HDUList([hdu,nres,sigh])
    hdulist.writeto(input_pars['output_dir']+input_pars['filename']+'_%s'%filt+input_pars['output_tag'],clobber=True)

