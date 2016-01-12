# fits lens light and arc simultaneously, using an optimizer to regulate the relative flux between lens and source

import pymc
import pyfits
import numpy as np
from photofit import convolve, indexTricks as iT
from pylens import pylens, MassModels, SBModels as models, plotting_tools
import pylab
import sys
#from plotters import cornerplot


lensname = sys.argv[1]
dpath = '/gdrive/working_dir/yattalens/specz_sample/cutout_dir/'
psfpath = '/gdrive/working_dir/yattalens/specz_sample/psf_dir/'

rmax = 30.
source_range = 8.
lens_range = 2.

bands = ['g', 'r', 'i']
allbands = ['g', 'r', 'i', 'z', 'y']

#bands = ['g']
lband = 'i'

zps = {'g': 28., 'r': 28., 'i': 28, 'z': 28, 'y': 28}

Nsamp = 10000
burnin = 0

x0 = 50.
y0 = 50.

sigmas = {}
images = {}
psfs = {}

for band in allbands:
    images[band] = pyfits.open(dpath+lensname+'_%s.fits'%band)[0].data.copy()
    sigmas[band] = pyfits.open(dpath+lensname+'_%s_var.fits'%band)[0].data.copy()**0.5
    psfs[band] = pyfits.open(psfpath+lensname+'_%s_psf.fits'%band)[1].data.copy()

Y, X = iT.coords(images[band].shape)

# makes a mask
mask = np.ones(images[band].shape, dtype=int)
dists = ((X - x0)**2 + (Y - y0)**2)**0.5
mask[dists > rmax] = 0
mask_r = mask.ravel() == 1


def visual_comparison(lens, lights, sources, images, sigmas, X, Y):
    colors = ['i', 'r', 'g']

    data_imgs = []
    model_imgs = []
    source_imgs = []
    for band in colors:
        data_imgs.append(images[band])

        logp, mags, modelimg = pylens.getModel(lens, lights[band], sources[band], images[band], sigmas[band], \
                                           X, Y, zp=zps[band], returnImg=True, mask=mask_r)

        model_imgs.append(modelimg[0] + modelimg[1])
        source_imgs.append(modelimg[1])

    plotting_tools.visual_comparison(data_imgs, model_imgs, source_imgs)

#lens parameters
x = pymc.Uniform('x', lower=x0-lens_range, upper=x0+lens_range, value=x0)
y = pymc.Uniform('y', lower=y0-lens_range, upper=y0+lens_range, value=y0)
rein = pymc.Uniform('rein', lower=0., upper=0.8*rmax, value=0.6*rmax)
pa = pymc.Uniform('pa', lower=-90., upper=180., value=70.)
q = pymc.Uniform('q', lower=0.3, upper=1., value=0.7)

#light parameters
reff = pymc.Uniform('reff', lower=1., upper=50., value=10.)
pas = pymc.Uniform('pas', lower=-90., upper=180., value=70.)
qs = pymc.Uniform('qs', lower=0.3, upper=2., value=0.7)
n = pymc.Uniform('n', lower=0.5, upper=8., value=4.)


#source parameters
sx = pymc.Uniform('sx', lower=x0-source_range, upper=x0+source_range, value=x0)
sy = pymc.Uniform('sy', lower=y0-source_range, upper=y0+source_range, value=y0)
sp = pymc.Uniform('sp', lower=-90., upper=180., value=0.)
sq = pymc.Uniform('sq', lower=0.3, upper=2., value=0.8)
sr = pymc.Uniform('sr', lower=0., upper=10., value=3.)
sn = pymc.Uniform('sn', lower=0.5, upper=8., value=1.)





lpars = [x, y, reff, pas, qs, n]
lcov = [0.01, 0.01, 1., 1., 0.01, 0.01]

lenspars = [x, y, rein, pa, q, sx, sy, sp, sq, sr, sn]
lenscov = [0.01, 0.01, 0.1, 1., 0.01, 1., 1., 1., 0.01, 0.1, 0.01]

pars = [x, y, rein, pa, q, reff, pas, qs, n, sx, sy, sp, sq, sr, sn]
cov = [0.01, 0.01, 0.1, 1., 0.01, 1., 1., 0.01, 0.01, 1., 1., 1., 0.01, 0.1, 0.01]


#defines the lens model
lens = MassModels.PowerLaw('lens', {'x':x, 'y':y, 'b':rein, 'q':q, 'pa':pa, 'eta':1.})

lights = {}
sources = {}

for band in allbands:
    light = models.Sersic('LensLight', {'x':x, 'y':y, 're':reff, 'q':qs, 'pa':pas, 'n':n})

    source = models.Sersic('src', {'x':sx, 'y':sy, 're':sr, 'q':sq, 'pa':sp, 'n':sn})

    light.convolve = convolve.convolve(images[band], psfs[band])[1]
    source.convolve = convolve.convolve(images[band], psfs[band])[1]

    lights[band] = light
    sources[band] = source


visual_comparison(lens, lights, sources, images, sigmas, X, Y)
pylab.show()

trace = pylens.do_fit_lightonly(lpars, lcov, [lband], lights, images, sigmas, X, Y, mask_r, zps, Nsamp=10000, burnin=0)

sx.value = x.value
sy.value = y.value
q.value = qs.value
pa.value = pas.value

for par in lpars:
    print str(par), par.value

visual_comparison(lens, lights, sources, images, sigmas, X, Y)
pylab.show()


"""
fewpars = [rein, sx, sy]
fewcov = [0.3, 1., 1.]

trace = pylens.do_fit(fewpars, fewcov, bands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, Nsamp=10000, burnin=0)
for par in fewpars:
    print str(par), par.value
"""

emceepars = [rein, q, pa, sx, sy]

trace = pylens.do_fit_emcee(emceepars, bands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, \
                            nwalkers=30, nsamp=100)
for par in pars:
    print str(par), par.value

visual_comparison(lens, lights, sources, images, sigmas, X, Y)
pylab.show()



trace = pylens.do_fit_emcee(pars, allbands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, \
                            usecov = np.ones(len(pars), dtype=bool), steps=np.array(cov)**0.5, nwalkers=30, nsamp=300)
for par in pars:
    print str(par), par.value

#visual_comparison(lens, lights, sources, images, sigmas, X, Y)
#pylab.show()

trace = pylens.do_fit(pars, cov, allbands, lens, lights, sources, images, sigmas, X, Y, mask_r, zps, Nsamp=Nsamp, burnin=0)



pylab.subplot(2, 1, 1)
pylab.plot(trace['logp'])
pylab.subplot(2, 1, 2)
pylab.plot(trace['rein'])
pylab.show()

#cornerplot(cp, color='r')
#pylab.show()

# saves everything
import pickle
#output = {'images': images, 'lens': lens, 'source': sources, 'light': lights, 'trace': trace, 'pars': pars}
#output = {'images': images, 'trace': trace, 'pars': pars}
f = open('chains/%s_pymc_chain.dat'%lensname, 'w')
pickle.dump(trace, f)
f.close()


ML = trace['logp'].argmax()
print ML
mlpars = []
mlscalepars = {}
for par in pars:
    mlval = trace[str(par)][ML]
    print str(par), mlval
    mlpars.append(mlval)
    par.value = mlval



model_imgs = {}
source_imgs = {}

for band in bands:
    logp, mags, mimage = pylens.getModel(lens, lights[band], sources[band], images[band], \
                                         sigmas[band], X, Y, zp=zps[band], returnImg=True, mask=mask_r)

    pyfits.PrimaryHDU(mimage[0]+mimage[1]).writeto('fitsfiles/%s_pymc_model_%s.fits'%(lensname,band), clobber=True)
    model_imgs[band] = mimage[0] + mimage[1]
    source_imgs[band] = mimage[1]

dlist = []
mlist = []
slist = []

rgbbands = ['i', 'r', 'g']
for band in rgbbands:
    dlist.append(images[band])
    mlist.append(model_imgs[band])
    slist.append(source_imgs[band])

plotting_tools.make_rgb_png(dlist, mlist, slist, 'figs/'+lensname+'_pymc_rgb.png')

visual_comparison(lens, lights, sources, images, sigmas, X, Y)
pylab.show()




