from yattaconfig import *
import numpy as np
import pyfits
import os
from photofit import indexTricks as iT, convolve
import pymc
from pylens import pylens, SBModels
from pylens import MassModels


"""
class Object:
    def __init__(self, ind=ind, x=None, y=None, r=r, npix=None, ab=None, flux=None, err=None, tang=None, arc=None):

        self.ind = ind
        self.x = x
        self.y = y
        self.r = r
        self.npix = npix
        self.ab = ab
        self.flux = flux
        self.err = err
        self.
"""

class light_model:

    def __init__(self, candidate):

        self.x = pymc.Uniform('x', lower=candidate.x0 - 2., upper=candidate.x0 + 2., value=candidate.x0)
        self.y = pymc.Uniform('y', lower=candidate.y0 - 2., upper=candidate.y0 + 2., value=candidate.y0)
        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.3, upper=2., value=0.7)
        self.re = pymc.Uniform('re', lower=3., upper=30., value=10.)
        self.n = pymc.Uniform('ns', lower=0.5, upper=8., value=4.)

        self.model = {}

        for band in candidate.bands:
            light = SBModels.Sersic('LensLight', {'x': self.x, \
                                                'y': self.y, \
                                                're': self.re, \
                                                'q': self.q, \
                                                'pa': self.pa, \
                                                'n': self.n})

            light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = light


class ring_model:

    def __init__(self, candidate):

        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.2, upper=2., value=0.6)
        self.rr = pymc.Uniform('rr', lower=0., upper=30., value=10.)
        self.hi = pymc.Uniform('hi', lower=1., upper=30., value=5.)
        self.ho = pymc.Uniform('ho', lower=1., upper=30., value=5.)

        self.model = {}

        for band in candidate.bands:
            ring = SBModels.Ring('Ring', {'x': candidate.x, \
                                          'y': candidate.y, \
                                          'rr': self.rr, \
                                          'q': self.q, \
                                          'pa': self.pa, \
                                          'hi': self.hi, \
                                          'ho': self.ho,})

            ring.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = ring


class new_ring_model:

    def __init__(self, candidate):

        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.2, upper=2., value=0.6)
        self.rr = pymc.Uniform('rr', lower=0., upper=30., value=10.)
        self.width = pymc.Uniform('width', lower=0., upper=5., value=3.)
        self.smooth = pymc.Uniform('smooth', lower=0., upper=10., value=1.)

        self.model = {}

        for band in candidate.bands:
            ring = SBModels.StoneRing('Ring', {'x': candidate.x, \
                                          'y': candidate.y, \
                                          'rr': self.rr, \
                                          'q': self.q, \
                                          'pa': self.pa, \
                                          'spa': 0., \
                                          'omega': 180., 'smooth': self.smooth, 'stone': 0., 'width': self.width})

            ring.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = ring


class lens_model:

    def __init__(self, candidate):

        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.2, upper=1., value=0.6)
        self.rein = pymc.Uniform('rein', lower=3., upper=30., value=10.)
        self.source_x = pymc.Uniform('hi', lower=candidate.x0 - source_range, upper=candidate.x0 + source_range, value=candidate.x0)
        self.source_y = pymc.Uniform('hi', lower=candidate.y0 - source_range, upper=candidate.y0 + source_range, value=candidate.y0)
        self.source_pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.source_q = pymc.Uniform('q', lower=0.2, upper=2., value=1.)
        self.source_re = pymc.Uniform('q', lower=0., upper=2., value=2.)

        self.lens = MassModels.PowerLaw('lens', {'x': candidate.x, 'y': candidate.y, 'b': self.rein, \
                                                       'pa': self.pa, 'q': self.q, 'eta': 1.})

        self.source = {}

        for band in candidate.bands:
            source = SBModels.Sersic('Source', {'x': self.source_x, 'y': self.source_y, 're': self.source_re, \
                                                'pa': self.source_pa, 'q': self.source_q, 'n': 1.})

            source.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.source[band] = source


class sersic_model:

    def __init__(self, candidate):

        self.x = pymc.Uniform('x', lower=candidate.x0 - 2., upper=candidate.x0 + 2., value=candidate.x0)
        self.y = pymc.Uniform('y', lower=candidate.y0 - 2., upper=candidate.y0 + 2., value=candidate.y0)
        self.pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
        self.q = pymc.Uniform('q', lower=0.1, upper=1., value=0.5)
        self.re = pymc.Uniform('re', lower=1., upper=20., value=5.)
        self.b4 = pymc.Uniform('b4', lower=-0.1, upper=0.1, value=0.)

        self.model = {}

        for band in candidate.bands:
            light = SBModels.Sersic_wboxyness('LensLight', {'x': self.x, 'y': self.y, 're': self.re, 'q': self.q, 'pa': self.pa, \
                                                  'n': 1., 'b4': self.b4})

            light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = light


class foreground_model:

    def __init__(self, candidate, objects, arcs):

        self.bad_arcs = None
        self.components = []
        self.amps = {}
        mask_all = np.ones(candidate.imshape, dtype=int)

        furthest = 0.
        for arc in arcs:
            mask_all[arc['footprint'] > 0] = 0
            if arc['r'] > furthest:
                furthest = arc['r']

        arc_mask = 1 - mask_all

        nobj = len(objects)

        fluxes = []
        inner_objects = []
        print 'foreground objects:'
        for obj in objects:
            mask_all[obj['footprint'] > 0] = 0
            if obj['r'] < modeluntil*furthest and obj['r'] > minarcdist and arc_mask[obj['y'] - 1, obj['x'] - 1] == 0:
                fluxes.append(candidate.sci[lightband][obj['footprint'] > 0].sum())
                inner_objects.append(obj)

        foregrounds = []

        if len(inner_objects) > 0:
            tmp = zip(fluxes, inner_objects)

            tmp.sort(reverse=True)

            fluxes, inner_objects = zip(*tmp)


            mask = mask_all.copy()

            for obj in inner_objects:

                component = {}

                xobj = obj['x']
                yobj = obj['y']

                remax = (obj['npix']/np.pi)**0.5

                x = pymc.Uniform('x', lower=xobj - 3., upper=xobj + 3., value=xobj)
                y = pymc.Uniform('y', lower=yobj - 3., upper=yobj + 3., value=yobj)

                pa = pymc.Uniform('pa', lower=-100., upper=100., value=0.)
                q = pymc.Uniform('q', lower=0.1, upper=2., value=0.7)
                re = pymc.Uniform('re', lower=1., upper=remax, value=min(5., remax))

                model = {}

                for band in candidate.bands:
                    light = SBModels.Sersic('LensLight', {'x': x, 'y': y, 're': re, 'q': q, 'pa': pa, \
                                                      'n': 4.})

                    light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

                    model[band] = light

                component['pars'] = [x, y, pa, q, re]
                component['model'] = model

                mask[obj['footprint'] > 0] = 1

                component['mask'] = mask.copy()
                component['dofit'] = True

                foregrounds.append(component)

        self.components = foregrounds

    def update(self, candidate, image_set):

        ncomp = len(self.components)

        for i in range(ncomp):
            xcomp = self.components[i]['pars'][0].value
            ycomp = self.components[i]['pars'][1].value

            dofit = True
            for image in image_set['images']:

                if image['footprint'][int(round(ycomp)), int(round(xcomp))] > 0:
                    dofit = False

            for arc in image_set['arcs']:
                if arc['footprint'][int(round(ycomp)), int(round(xcomp))] > 0:
                    dofit = False

            if dofit:
                print 'foreground %d at x: %2.1f y: %2.1f is modeled'%(i+1, xcomp, ycomp)
                self.components[i]['dofit'] = True
            else:
                self.components[i]['dofit'] = False
                print 'foreground %d at x: %2.1f y: %2.1f is not included'%(i+1, xcomp, ycomp)

        bad_arcs = []

        for arc in image_set['bad_arcs']:

            component = {}

            xarc = arc['x']
            yarc = arc['y']

            ang = np.arctan((yarc - candidate.y)/(xarc - candidate.x))

            if xarc < candidate.x:
                ang += np.pi

            ang *= 180./np.pi

            remax = (arc['npix']/np.pi*arc['ab'])**0.5

            x = pymc.Uniform('x', lower=xarc - 3., upper=xarc + 3., value=xarc)
            y = pymc.Uniform('y', lower=yarc - 3., upper=yarc + 3., value=yarc)

            invrc = pymc.Uniform('invrc', lower=-0.2, upper=0.2, value=0.0001)
            pa = pymc.Uniform('pa', lower=ang - 30., upper=ang + 30., value=ang)
            length = pymc.Uniform('length', lower=0., upper=remax, value=0.5*remax)
            h = pymc.Uniform('h', lower=0., upper=5., value=2.)

            model = {}

            for band in candidate.bands:
                light = SBModels.Arc('Arc', {'x': x, 'y': y, 'invrc': invrc, 'length': length, 'pa': pa, 'hr': h, \
                                             'ht': h})

                light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

                model[band] = light

            component['pars'] = [x, y, pa, invrc, length, h]
            component['model'] = model

            bad_arcs.append(component)

        self.bad_arcs = bad_arcs


class Candidate:

    def __init__(self, name, bands=('i', 'r', 'g'), zp=(27., 27., 27.)):
        self.name = name
        self.bands = bands

        self.zp = {}
        i=0
        for band in self.bands:
            self.zp[band] = zp[i]

        self.x0 = None
        self.y0 = None
        self.X = None
        self.Y = None
        self.R = None
        self.sci = {}
        self.var = {}
        self.err = {}
        self.psf = {}
        self.imshape = None
        self.dataok = None

        self.x = None
        self.y = None
        self.light_pa = None
        self.light_re = None
        self.light_q = None
        self.light_n = None

        self.lens_pa = None
        self.lens_q = None
        self.lens_rein = None

        self.source_x = None
        self.source_y = None
        self.source_re = None
        self.source_pa = None
        self.source_q = None
        self.soruce_re = None

        self.ring_pa = None
        self.ring_q = None
        self.ring_rr = None
        self.ring_hi = None
        self.ring_ho = None
        self.ring_smooth = None
        self.ring_width = None

        self.sersic_x = None
        self.sersic_y = None
        self.sersic_re = None
        self.sersic_pa = None
        self.sersic_q = None
        self.sersic_b4 = None

        self.light_pars_sample = None
        self.lens_pars_sample = None
        self.ring_pars_sample = None
        self.sersic_pars_sample = None

        self.lenssub_model = {}
        self.lenssub_resid = {}
        self.foreground_model = {}
        self.ringfit_model = {}
        self.lensfit_model = {}
        self.sersicfit_model = {}

        self.lensfit_mask = None
        self.ringfit_mask = None
        self.sersicfit_mask = None

        self.ringfit_chi2 = None
        self.lensfit_chi2 = None
        self.sersicfit_chi2 = None

        self.image_sets = None

        self.source_footprint = None

        self.ringfit_footprint_chi2 = None
        self.lensfit_footprint_chi2 = None
        self.sersicfit_footprint_chi2 = None

	self.footprint_rms = None
	self.sextractor_rms = None

        self.model_angular_aperture = None

    def read_data(self):

        print 'reading in data...'

        found = True
        shapeok = True

        for band in self.bands:
            sciname = datadir+'/%s_%s_sci.fits' % (self.name, band)
            psfname = datadir+'/%s_%s_psf.fits' % (self.name, band)
            varname = datadir+'/%s_%s_var.fits' % (self.name, band)

            if os.path.isfile(sciname) and os.path.isfile(psfname) and os.path.isfile(varname):
                sci = pyfits.open(sciname)[0].data.copy()
                var = pyfits.open(varname)[0].data.copy()
                psf = pyfits.open(psfname)[1].data.copy()

                self.sci[band] = sci
                self.var[band] = var
                self.err[band] = var**0.5
                self.psf[band] = psf

                s = sci.shape

                if self.imshape is None:
                    self.imshape = s
                else:
                    if s != self.imshape:
                        shapeok = False

            else:
                found = False

        if found and shapeok:
            self.x0 = float(self.imshape[1]/2)
            self.y0 = float(self.imshape[0]/2)

            Y, X = iT.coords(self.imshape)

            self.Y = Y
            self.X = X

            R = ((X - self.x0)**2 + (Y - self.y0)**2)**0.5
            self.R = R

            self.dataok = True

        else:
            self.dataok = False

        return self.dataok

    def get_source_footprint(self, nsig=2.):

        footprint = np.zeros(self.imshape, dtype=int)

        for band in fitband:
            sdet = self.lensfit_model[band][-1] > nsig*self.err[band]

            footprint[sdet] = 1

        self.source_footprint = footprint

    def get_footprint_chi2(self, image_set):

        lchi2 = 0.
        #rchi2 = 0.
        #cchi2 = 0.

        mask = self.source_footprint

        for junk in image_set['junk']:
            mask[junk['footprint'] > 0] = 0

        mask[self.R > maxarcdist] = 0

        for band in fitband:
            lchi2 += (((self.sci[band] - self.lensfit_model[band][0] - self.lensfit_model[band][1])/self.err[band])**2)[mask > 0].sum()
            #rchi2 += (((self.sci[band] - self.ringfit_model[band][0] - self.ringfit_model[band][1])/self.err[band])**2)[mask > 0].sum()
            #rchi2 += (((self.sci[band] - self.sersicfit_model[band][0] - self.sersicfit_model[band][1])/self.err[band])**2)[mask > 0].sum()

        self.lensfit_footprint_chi2 = lchi2
        #self.ringfit_footprint_chi2 = rchi2
        #self.sersicfit_footprint_chi2 = rchi2

    def get_footprint_rms(self, image_set):

        rms = 0.

        mask = self.source_footprint

        for junk in image_set['junk']:
            mask[junk['footprint'] > 0] = 0

        mask[self.R > maxarcdist] = 0

        for band in fitband:
            medflux = np.median(self.sci[band][mask > 0])

            modflux = 0.*self.sci[band]
            for comp in self.lensfit_model[band]:
                modflux += comp

            rms += (((self.sci[band] - modflux)**2/medflux**2)[mask > 0].sum() / float(mask.sum()))**0.5

        self.footprint_rms = rms

    def get_sextractor_rms(self, image_set):

        rms = 0.

        mask = np.zeros(self.imshape, dtype=int)

        for arc in image_set['arcs']:
            mask[arc['footprint'] > 0] = 1

        for band in fitband:
            medflux = np.median(self.sci[band][mask > 0])

            modflux = 0.*self.sci[band]
            for comp in self.lensfit_model[band]:
                modflux += comp

            rms += (((self.sci[band] - modflux)**2/medflux**2)[mask > 0].sum() / float(mask.sum()))**0.5

        self.sextractor_rms = rms

    def get_model_angular_aperture(self):

        ypix = self.Y[self.source_footprint > 0]
        xpix = self.X[self.source_footprint > 0]
        rpix = ((xpix - self.x)**2 + (ypix - self.y)**2)**0.5
        cospix = (xpix - self.x)/rpix
        sinpix = (ypix - self.y)/rpix

        npix = len(xpix)

        max_aperture = 0.
        for j in range(npix):
            cosdiff = cospix[j]*cospix + sinpix[j]*sinpix
            aperture = 180.*np.arccos(cosdiff).max()/np.pi
            if aperture > max_aperture:
                max_aperture = aperture

        self.model_angular_aperture = max_aperture
