from yattaconfig import def_config
import numpy as np
from astropy.io import fits as pyfits
import os
from photofit import indexTricks as iT, convolve
from yattapylens import pylens, SBModels
from yattapylens import MassModels


class YattaPar:

    def __init__(self, name, lower=0., upper=1., value=0.):

        self.name = name
        self.lower = lower
        self.upper = upper
        self.value = value


class light_model:

    def __init__(self, candidate, config=def_config):

        self.x = YattaPar('x', lower=candidate.x0 - 2., upper=candidate.x0 + 2., value=candidate.x0)
        self.y = YattaPar('y', lower=candidate.y0 - 2., upper=candidate.y0 + 2., value=candidate.y0)
        self.pa = YattaPar('pa', lower=-100., upper=100., value=0.)
        self.q = YattaPar('q', lower=0.2, upper=2., value=0.8)
        self.re = YattaPar('re', lower=3., upper=config['lightfit_reff_max'], value=10.)
        self.n = YattaPar('ns', lower=config['lightfit_nser_min'], upper=config['lightfit_nser_max'], value=4.)

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

    def __init__(self, candidate, config=def_config):

        self.pa = YattaPar('pa', lower=-100., upper=100., value=0.)
        self.q = YattaPar('q', lower=0.2, upper=2., value=0.8)
        self.rr = YattaPar('rr', lower=0., upper=config['maxarcdist'], value=0.3*config['maxarcdist'])
        self.hi = YattaPar('hi', lower=1., upper=30., value=5.)
        self.ho = YattaPar('ho', lower=1., upper=30., value=5.)

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

        self.pa = YattaPar('pa', lower=-100., upper=100., value=0.)
        self.q = YattaPar('q', lower=0.2, upper=2., value=0.6)
        self.rr = YattaPar('rr', lower=0., upper=30., value=10.)
        self.width = YattaPar('width', lower=0., upper=5., value=3.)
        self.smooth = YattaPar('smooth', lower=0., upper=10., value=1.)

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

    def __init__(self, candidate, config=def_config):

        self.pa = YattaPar('pa', lower=-100., upper=100., value=0.)
        self.q = YattaPar('q', lower=0.2, upper=1., value=0.6)
        self.rein = YattaPar('rein', lower=3., upper=30., value=10.)
        self.source_x = YattaPar('hi', lower=candidate.x0 - config['source_range'], upper=candidate.x0 + config['source_range'], value=candidate.x0)
        self.source_y = YattaPar('hi', lower=candidate.y0 - config['source_range'], upper=candidate.y0 + config['source_range'], value=candidate.y0)
        self.source_pa = YattaPar('pa', lower=-100., upper=100., value=0.)
        self.source_q = YattaPar('q', lower=0.2, upper=2., value=1.)
        self.source_re = YattaPar('q', lower=0., upper=2., value=2.)

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

        self.x = YattaPar('x', lower=candidate.x0 - 2., upper=candidate.x0 + 2., value=candidate.x0)
        self.y = YattaPar('y', lower=candidate.y0 - 2., upper=candidate.y0 + 2., value=candidate.y0)
        self.pa = YattaPar('pa', lower=-100., upper=100., value=0.)
        self.q = YattaPar('q', lower=0.1, upper=1., value=0.5)
        self.re = YattaPar('re', lower=1., upper=20., value=5.)
        self.b4 = YattaPar('b4', lower=-0.1, upper=0.1, value=0.)

        self.model = {}

        for band in candidate.bands:
            light = SBModels.Sersic_wboxyness('LensLight', {'x': self.x, 'y': self.y, 're': self.re, 'q': self.q, 'pa': self.pa, \
                                                  'n': 1., 'b4': self.b4})

            light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

            self.model[band] = light


class foreground_model:

    def __init__(self, candidate, objects, arcs, config=def_config):

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
        print('foreground objects:')
        for obj in objects:
            mask_all[obj['footprint'] > 0] = 0
            if obj['r'] < config['modelmaxdist']*furthest and obj['r'] > config['minarcdist'] and \
                            arc_mask[int(round(obj['y'])) - 1, int(round(obj['x'])) - 1] == 0:
                fluxes.append(candidate.sci[config['lightband']][obj['footprint'] > 0].sum())
                inner_objects.append(obj)

        foregrounds = []

        if len(inner_objects) > 0:
            tmp = zip(fluxes, inner_objects)

            fluxes, inner_objects = zip(*sorted(tmp, reverse=True))


            mask = mask_all.copy()

            for obj in inner_objects:

                component = {}

                xobj = obj['x']
                yobj = obj['y']

                remax = (obj['npix']/np.pi)**0.5

                x = YattaPar('x', lower=xobj - 3., upper=xobj + 3., value=xobj)
                y = YattaPar('y', lower=yobj - 3., upper=yobj + 3., value=yobj)

                pa = YattaPar('pa', lower=obj['pa']-100., upper=obj['pa']+100., value=obj['pa'])
                q = YattaPar('q', lower=0.1, upper=2., value=max(0.1, 1./obj['ab']))
                re = YattaPar('re', lower=1., upper=remax, value=min(5., remax))

                model = {}

                for band in candidate.bands:
                    light = SBModels.Sersic('foreground', {'x': x, 'y': y, 're': re, 'q': q, 'pa': pa, \
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
            if int(round(ycomp)) >= 0 and int(round(ycomp)) < candidate.imshape[0] and int(round(xcomp)) > 0 and int(round(xcomp)) < candidate.imshape[1]:
                for image in image_set['images']:
                    if image['footprint'][int(round(ycomp)), int(round(xcomp))] > 0:
                        dofit = False

                for arc in image_set['arcs']:
                    if arc['footprint'][int(round(ycomp)), int(round(xcomp))] > 0:
                        dofit = False
            else:
                dofit = False

            if dofit:
                print('foreground %d at x: %2.1f y: %2.1f is modeled'%(i+1, xcomp, ycomp))
                self.components[i]['dofit'] = True
            else:
                self.components[i]['dofit'] = False
                print('foreground %d at x: %2.1f y: %2.1f is not included'%(i+1, xcomp, ycomp))

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

            x = YattaPar('x', lower=xarc - 3., upper=xarc + 3., value=xarc)
            y = YattaPar('y', lower=yarc - 3., upper=yarc + 3., value=yarc)

            invrc = YattaPar('invrc', lower=-0.2, upper=0.2, value=0.0001)
            pa = YattaPar('pa', lower=ang - 30., upper=ang + 30., value=ang)
            length = YattaPar('length', lower=0., upper=remax, value=0.5*remax)
            h = YattaPar('h', lower=0., upper=5., value=2.)

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

        # adds any foreground detected in the g band that wasn't detected in i, if any,
        # or objects that were thought to be arcs but are not.

        new_foregrounds = []

        for obj in image_set['foregrounds']:

            already_there = False

            for i in range(ncomp):
                xcomp = self.components[i]['pars'][0].value
                ycomp = self.components[i]['pars'][1].value

                if obj['footprint'][int(round(ycomp)), int(round(xcomp))] > 0:
                    already_there = True

            if not already_there:
                component = {}

                xobj = obj['x']
                yobj = obj['y']

                print('adding foreground at %2.1f %2.1f'%(xobj, yobj))

                remax = (obj['npix']/np.pi)**0.5

                x = YattaPar('x', lower=xobj - 3., upper=xobj + 3., value=xobj)
                y = YattaPar('y', lower=yobj - 3., upper=yobj + 3., value=yobj)

                pa = YattaPar('pa', lower=obj['pa']-100., upper=obj['pa']+100., value=obj['pa'])
                q = YattaPar('q', lower=0.1, upper=2., value=max(0.1, 1./obj['ab']))
                re = YattaPar('re', lower=1., upper=remax, value=min(5., remax))

                model = {}

                for band in candidate.bands:
                    light = SBModels.Sersic('LensLight', {'x': x, 'y': y, 're': re, 'q': q, 'pa': pa, \
                                                      'n': 4.})

                    light.convolve = convolve.convolve(candidate.sci[band], candidate.psf[band])[1]

                    model[band] = light

                component['pars'] = [x, y, pa, q, re]
                component['model'] = model

                new_foregrounds.append(component)

        self.new_foregrounds = new_foregrounds


class Candidate:

    def __init__(self, name, bands=('i', 'r', 'g'), zp=(27., 27., 27.), ra=None, dec=None, config=def_config):
        self.name = name
        self.bands = bands

        self.zp = {}
        i=0
        for band in self.bands:
            self.zp[band] = zp[i]

        self.fitband = config['fitband']
        self.lightband = config['lightband']

        self.config = config.copy()

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

        self.ra = ra
        self.dec = dec

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
        self.lens_mags_sample = None
        self.ring_pars_sample = None
        self.sersic_pars_sample = None

        self.lenssub_model = {}
        self.lenssub_resid = {}
        self.foreground_model = {}
        self.ringfit_model = {}
        self.lensfit_model = {}
        self.sersicfit_model = {}

        self.lensfit_mags = {}

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

    def read_data(self, data_dir=None):

        print('reading in data...')

        found = True
        shapeok = True

        if data_dir is None:
            data_dir = self.config['datadir']

        for band in self.bands:
            sciname = data_dir+'/%s_%s_sci.fits' % (self.name, band)
            psfname = data_dir+'/%s_%s_psf.fits' % (self.name, band)
            varname = data_dir+'/%s_%s_var.fits' % (self.name, band)

            if os.path.isfile(sciname) and os.path.isfile(psfname) and os.path.isfile(varname):
                sci = pyfits.open(sciname)[0].data.copy()
                var = pyfits.open(varname)[0].data.copy()
                hdulist = pyfits.open(psfname)
                if hdulist[0].data is None:
                    psf = hdulist[1].data.copy()
                else:
                    psf = hdulist[0].data.copy()

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

        for band in self.fitband:
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

        mask[self.R > self.config['maxarcdist']] = 0

        for band in self.fitband:
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

        mask[self.R > self.config['maxarcdist']] = 0

        for band in self.fitband:
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

        for band in self.fitband:
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

    def save_model(self, outname=None, imset=0, clobber=False):

        if outname is None:
            outname = '%s_model.fits'%self.name

        hdr = pyfits.Header()

        hdr['LENS_X'] = self.x
        hdr['LENS_Y'] = self.y
        hdr['LIGHT_PA'] = self.light_pa
        hdr['LIGHT_Q'] = self.light_q
        hdr['LIGHT_RE'] = self.light_re
        hdr['LIGHT_N'] = self.light_n

        hdr['LENS_PA'] = self.lens_pa
        hdr['LENS_Q'] = self.lens_q
        hdr['REIN'] = self.lens_rein
        hdr['S_X'] = self.source_x
        hdr['S_Y'] = self.source_y
        hdr['S_RE'] = self.source_re

        for band in self.bands:
            if band in self.lensfit_mags:
                lmag = self.lensfit_mags[band][0]
                smag = self.lensfit_mags[band][-1]
    
                if not np.isfinite(lmag):
                    lmag = 99.
                if not np.isfinite(smag):
                    smag = 99.
            else:
                lmag = None
                smag = None

            hdr['LMAG_%s'%band] = lmag
            hdr['SMAG_%s'%band] = smag

        hdr['LENSCHI2'] = self.lensfit_chi2
        hdr['RINGCHI2'] = self.ringfit_chi2
        hdr['SERCHI2'] = self.sersicfit_chi2

        hdr['ANG_AP'] = self.model_angular_aperture

        if self.image_sets is not None:
            image_set = self.image_sets[imset]
    
            narcs = len(image_set['arcs'])
            hdr['NARCS'] = narcs
    
            ncim = len(image_set['images'])
            hdr['NCIM'] = ncim
    
            njunk = len(image_set['junk'])
            hdr['NJNK'] = njunk
    
            nfgd = len(image_set['foregrounds'])
            nbad = len(image_set['bad_arcs'])
            hdr['NFGD'] = nfgd + nbad
    
            arc_segmap = np.zeros(self.imshape, dtype=int)
    
            for i in range(narcs):
                hdr['ARC%d_X'%(i+1)] = image_set['arcs'][i]['x']
                hdr['ARC%d_Y'%(i+1)] = image_set['arcs'][i]['y']
                hdr['ARC%d_R'%(i+1)] = image_set['arcs'][i]['r']
                hdr['ARC%d_NPX'%(i+1)] = image_set['arcs'][i]['npix']
                hdr['ARC%d_PA'%(i+1)] = image_set['arcs'][i]['pa']
                hdr['ARC%d_DPA'%(i+1)] = image_set['arcs'][i]['ang_diff']
                hdr['ARC%d_AB'%(i+1)] = image_set['arcs'][i]['ab']
    
                arc_segmap[image_set['arcs'][i]['footprint'] > 0] = i+1
    
            cim_segmap = np.zeros(self.imshape, dtype=int)
    
            for i in range(ncim):
                hdr['CIM%d_X'%(i+1)] = image_set['images'][i]['x']
                hdr['CIM%d_Y'%(i+1)] = image_set['images'][i]['y']
                hdr['CIM%d_R'%(i+1)] = image_set['images'][i]['r']
                hdr['CIM%d_NPX'%(i+1)] = image_set['images'][i]['npix']
                hdr['CIM%d_PA'%(i+1)] = image_set['images'][i]['pa']
                hdr['CIM%d_DPA'%(i+1)] = image_set['images'][i]['ang_diff']
                hdr['CIM%d_AB'%(i+1)] = image_set['images'][i]['ab']
    
                cim_segmap[image_set['images'][i]['footprint'] > 0] = i+1
    
            junk_segmap = np.zeros(self.imshape, dtype=int)
    
            for i in range(njunk):
                hdr['JNK%d_X'%(i+1)] = image_set['junk'][i]['x']
                hdr['JNK%d_Y'%(i+1)] = image_set['junk'][i]['y']
                hdr['JNK%d_R'%(i+1)] = image_set['junk'][i]['r']
                hdr['JNK%d_NPX'%(i+1)] = image_set['junk'][i]['npix']
                hdr['JNK%d_PA'%(i+1)] = image_set['junk'][i]['pa']
                hdr['JNK%d_DPA'%(i+1)] = image_set['junk'][i]['ang_diff']
                hdr['JNK%d_AB'%(i+1)] = image_set['junk'][i]['ab']
    
                junk_segmap[image_set['junk'][i]['footprint'] > 0] = i+1
    
            fgd_segmap = np.zeros(self.imshape, dtype=int)
    
            for i in range(nfgd):
                hdr['FGD%d_X'%(i+1)] = image_set['foregrounds'][i]['x']
                hdr['FGD%d_Y'%(i+1)] = image_set['foregrounds'][i]['y']
                hdr['FGD%d_R'%(i+1)] = image_set['foregrounds'][i]['r']
                hdr['FGD%d_NPX'%(i+1)] = image_set['foregrounds'][i]['npix']
                hdr['FGD%d_PA'%(i+1)] = image_set['foregrounds'][i]['pa']
                hdr['FGD%d_DPA'%(i+1)] = image_set['foregrounds'][i]['ang_diff']
                hdr['FGD%d_AB'%(i+1)] = image_set['foregrounds'][i]['ab']
    
                fgd_segmap[image_set['foregrounds'][i]['footprint'] > 0] = i+1
    
            for i in range(nbad):
                hdr['FGD%d_X'%(nfgd+i+1)] = image_set['bad_arcs'][i]['x']
                hdr['FGD%d_Y'%(nfgd+i+1)] = image_set['bad_arcs'][i]['y']
                hdr['FGD%d_R'%(nfgd+i+1)] = image_set['bad_arcs'][i]['r']
                hdr['FGD%d_NPX'%(nfgd+i+1)] = image_set['bad_arcs'][i]['npix']
                hdr['FGD%d_PA'%(nfgd+i+1)] = image_set['bad_arcs'][i]['pa']
                hdr['FGD%d_DPA'%(nfgd+i+1)] = image_set['bad_arcs'][i]['ang_diff']
                hdr['FGD%d_AB'%(nfgd+i+1)] = image_set['bad_arcs'][i]['ab']
    
                fgd_segmap[image_set['bad_arcs'][i]['footprint'] > 0] = nfgd+i+1

        phdu = pyfits.PrimaryHDU(header=hdr)

        hdulist = pyfits.HDUList([phdu])

        for band in self.bands:

            sci_hdu = pyfits.ImageHDU(data=self.sci[band])
            sci_hdu.header['EXTNAME'] = 'SCI_%s'%band
            hdulist.append(sci_hdu)

            if band in self.lensfit_model:
                ncomp = len(self.lensfit_model[band])

                lens_full = 0. * self.sci[band]
    
                for i in range(ncomp):
                    lens_full += self.lensfit_model[band][i].copy()
    
                full_hdu = pyfits.ImageHDU(data=lens_full)
                full_hdu.header['EXTNAME'] = 'ALL_%s'%band
                hdulist.append(full_hdu)
    
                source_hdu = pyfits.ImageHDU(data=self.lensfit_model[band][-1])
                source_hdu.header['EXTNAME'] = 'SOURCE_%s'%band
                hdulist.append(source_hdu)

            if band in self.lenssub_model:
                lens_hdu = pyfits.ImageHDU(data=self.lenssub_model[band][0])
                lens_hdu.header['EXTNAME'] = 'LENS_%s'%band
                hdulist.append(lens_hdu)

        if self.image_sets is not None:
            arc_hdu = pyfits.ImageHDU(data=arc_segmap)
            arc_hdu.header['EXTNAME'] = 'ARC_SEG'
    
            cim_hdu = pyfits.ImageHDU(data=cim_segmap)
            cim_hdu.header['EXTNAME'] = 'CIM_SEG'
    
            fgd_hdu = pyfits.ImageHDU(data=fgd_segmap)
            fgd_hdu.header['EXTNAME'] = 'FGD_SEG'
    
            junk_hdu = pyfits.ImageHDU(data=junk_segmap)
            junk_hdu.header['EXTNAME'] = 'JNK_SEG'
    
            hdulist.append(arc_hdu)
            hdulist.append(cim_hdu)
            hdulist.append(fgd_hdu)
            hdulist.append(junk_hdu)

        hdulist.writeto(outname, clobber=clobber)

