from yattaconfig import *
import numpy as np
import sys
import yattaobjects as yo
from pylens import plotting_tools
import object_finder_tools as oft
import pickle
import fitters as fitters
import time
import os

tstart = time.clock()

name = sys.argv[1]

cand = yo.Candidate(name=name, bands=('i', 'r', 'g'))

loglines = []
logfile = open(logdir+name+'.txt', 'a')

if cand.read_data():

    light_model = yo.light_model(cand)

    lenspars, junkmask = oft.find_lens(cand, detect_band='i', detect_thresh=3.)

    if lenspars is None:
        quit('No galaxies found. Quitting YattaLens.')

    tlenssub_start = time.clock()

    #fitters.quick_lens_subtraction(cand, light_model, rmax=20., lfitband=(lightband), niter=200)
    guess = [lenspars['x'], lenspars['y'], lenspars['pa'], 1./lenspars['ab'], lenspars['npix']**0.5/np.pi, 4.]
    fitters.fit_light(cand, light_model, rmax=20., lfitband=(lightband), mask=junkmask, guess=guess, nsamp=50)

    tlenssub_end = time.clock()
    loglines.append('QUICK_SUBTRACTION_TIME %f\n'%(tlenssub_end - tlenssub_start))

    objects, arcs, segmap, foundarcs = oft.find_objects(cand, detect_band='g', detect_thresh=3.)
    tphase1_end = time.clock()
    loglines.append('PHASE_1_TIME %f\n'%(tphase1_end - tstart))
    loglines.append('ARC_CANDIDATES %d\n'%(len(arcs)))

    if foundarcs:
        iobjs, iarcs, junkmask, ifoundarcs = oft.find_objects(cand, detect_band=lightband, detect_thresh=3.)
        junkmask[junkmask > 0] = 1
        junkmask = 1 - junkmask
        junkmask[cand.R < 5.] = 1

        print 'arcs found: %d'%(len(arcs))

        guess = [cand.x, cand.y, cand.light_pa, cand.light_q, cand.light_re, cand.light_n]
        fitters.fit_light(cand, light_model, lfitband=(lightband), mask=junkmask, guess=guess, nsamp=200, rmax=20.)

        foreground_model = yo.foreground_model(cand, iobjs + iarcs, arcs)

        fitters.fit_foregrounds_fixedamps(cand, foreground_model, light_model)

        objects = oft.measure_fluxes(objects, cand, foreground_model)
        arcs = oft.measure_fluxes(arcs, cand, foreground_model)

        nobjs = len(objects)

        cand.image_sets = oft.determine_image_sets(objects, arcs)

        nsets = len(cand.image_sets)

        print 'possible image sets: %d'%nsets

        lens_model = yo.lens_model(cand)
        ring_model = yo.ring_model(cand)
        sersic_model = yo.sersic_model(cand)

        for i in range(nsets):

            bluest = np.inf
            for arc in cand.image_sets[i]['arcs']:
                ratio = (arc['i_flux'] - 3.*arc['i_err'])/(arc['g_flux'] + 3.*arc['g_err'])
            if ratio < bluest:
                bluest = ratio

            if bluest < 10.**(gmi_max/2.5):

                print 'set %d: %d arcs, %d images'%(i+1, len(cand.image_sets[i]['arcs']), \
                                                    len(cand.image_sets[i]['images']))

                for arc in cand.image_sets[i]['arcs']:
                    print 'arc', arc['x'], arc['y']
                for image in cand.image_sets[i]['images']:
                    print 'image', image['x'], image['y']

                foreground_model.update(cand, cand.image_sets[i])

                if len(foreground_model.bad_arcs) > 0:
                    fitters.fit_bad_arcs(cand, foreground_model, light_model)

                fitters.fit_lens(cand, lens_model, light_model, foreground_model, cand.image_sets[i])

                cand.get_model_angular_aperture()

                loglines.append('LENS_MODEL_ANGULAR_APERTURE %2.1f\n'%cand.model_angular_aperture)

                success = False

                if cand.model_angular_aperture > min_aperture:

                    if len(arcs) > 1:
                        fitters.fit_ring(cand, ring_model, light_model, foreground_model, cand.image_sets[i])
                        if cand.lensfit_chi2 < cand.ringfit_chi2:
                            fitters.fit_sersic(cand, sersic_model, light_model, foreground_model, cand.image_sets[i])
                            if cand.lensfit_chi2 < cand.sersicfit_chi2:
                                success = True
                    else:
                        fitters.fit_sersic(cand, sersic_model, light_model, foreground_model, cand.image_sets[i])
                        if cand.lensfit_chi2 < cand.sersicfit_chi2:
                            fitters.fit_ring(cand, ring_model, light_model, foreground_model, cand.image_sets[i])
                            if cand.lensfit_chi2 < cand.ringfit_chi2:
                                success = True

                if success:
                    cpdir = 'success'
                    print 'Yatta!'
                else:
                    print 'failed'
                    cpdir = 'failure'

                figname = 'figs/%s_imset%d.png'%(cand.name, i+1)

                plotting_tools.make_full_rgb(cand, cand.image_sets[i], outname=figname, success=success)

                #os.system('cp %s %s/'%(figname, cpdir))

            f = open(modeldir+'/%s_model_set%d.dat'%(cand.name, i+1), 'w')
            pickle.dump(cand, f)
            f.close()

        tphase2_end = time.clock()
        loglines.append('PHASE_2_TIME %f\n'%(tphase2_end - tphase1_end))

tend = time.clock()
loglines.append('TOTAL_TIME %f\n'%(tend - tstart))
logfile.writelines(loglines)
logfile.close()

