import yattaconfig
import numpy as np
import sys
import yattaobjects as yo
import yattargbtools
import object_finder_tools as oft
import yattafitters as fitters
import time
import os


tstart = time.time()

yattaconfig.write_config_file()
if len(sys.argv) > 1:
    configname = sys.argv[1]
else:
    configname = 'default.yatta'

config = yattaconfig.read_config_file(configname)
for par in config:
    print(par, config[par])

allbands = list(config['rgbbands'])
if not config['fitband'] in allbands:
    allbands.append(config['fitband'])
if not config['lightband'] in allbands:
    allbands.append(config['lightband'])

if os.path.isfile(config['catalog_file']):
    f = open(config['catalog_file'], 'r')
    input_lines = f.readlines()
    f.close()

else:
    input_lines = [config['catalog_file']]

summary_lines = []
#summary_lines.append('# Target_name\tSuccess\tReason\tbestmodel_no\tdata_flag\n')
for input_line in input_lines:

    name = input_line.split()[0]
    summary_line = input_line.rstrip()

    cand = yo.Candidate(name=name, bands=allbands, config=config)

    loglines = []

    isalens = False
    reason = ''

    lensness = []
    lenssets = []

    if cand.read_data():

        if config['expected_size'] is not None:
            if cand.imshape[0] == config['expected_size'] and cand.imshape[1] == config['expected_size']:
                data_flag = 0
            else:
                data_flag = 2
        else:
            data_flag = 0

        light_model = yo.light_model(cand, config=config)

        lenspars, junkmask = oft.find_lens(cand, detect_band=config['lightband'], detect_thresh=3., config=config)

        if lenspars is not None and junkmask[cand.R < config['lightfitrmax']].sum() > 0:

            tlenssub_start = time.time()

            guess = [lenspars['x'], lenspars['y'], lenspars['pa'], 1./lenspars['ab'], lenspars['npix']**0.5/np.pi, 4.]
            fitters.fit_light(cand, light_model, rmax=config['lightfitrmax'], lfitband=config['lightband'], mask=junkmask, guess=guess, \
                              nsamp=50, fit_method=config['lightfit_method'])

            tlenssub_end = time.time()
            loglines.append('QUICK_SUBTRACTION_TIME %f\n'%(tlenssub_end - tlenssub_start))

            objects, arcs, segmap, foundarcs = oft.find_objects(cand, detect_band=config['fitband'], detect_thresh=3., \
                                                                config=config)
            tphase1_end = time.time()
            loglines.append('PHASE_1_TIME %f\n'%(tphase1_end - tstart))
            loglines.append('ARC_CANDIDATES %d\n'%(len(arcs)))

            if foundarcs:
                iobjs, iarcs, junkmask, ifoundarcs = oft.find_objects(cand, detect_band=config['lightband'], detect_thresh=3., \
                                                                      config=config)
                junkmask[junkmask > 0] = 1
                junkmask = 1 - junkmask
                junkmask[cand.R < 5.] = 1

                print('arcs found: %d'%(len(arcs)))

                guess = [cand.x, cand.y, cand.light_pa, cand.light_q, cand.light_re, cand.light_n]
                fitters.fit_light(cand, light_model, lfitband=config['lightband'], mask=junkmask, guess=guess, nsamp=200, \
                                  rmax=config['lightfitrmax'], fit_method=config['lightfit_method'])

                foreground_model = yo.foreground_model(cand, iobjs + iarcs, arcs)

                fitters.fit_foregrounds_fixedamps(cand, foreground_model, light_model)

                objects = oft.measure_fluxes(objects, cand, foreground_model, \
                                             meas_bands=(config['fitband'], config['lightband']))

                arcs = oft.measure_fluxes(arcs, cand, foreground_model, \
                                          meas_bands=(config['fitband'], config['lightband']))

                nobjs = len(objects)

                cand.image_sets = oft.determine_image_sets(objects, arcs, iobjs + iarcs, config=config)

                nsets = len(cand.image_sets)

                print('possible image sets: %d'%nsets)

                lens_model = yo.lens_model(cand, config=config)
                ring_model = yo.ring_model(cand, config=config)
                sersic_model = yo.sersic_model(cand)

                for i in range(nsets):

                    figname = config['figdir']+'/%s_imset%d.png'%(cand.name, i+1)

                    bluest = np.inf
                    for arc in cand.image_sets[i]['arcs']:
                        ratio = (arc['%s_flux'%config['lightband']] - 3.*arc['%s_err'%config['lightband']])/\
                                (arc['%s_flux'%config['fitband']] + 3.*arc['%s_err'%config['fitband']])
                        if ratio < bluest:
                            bluest = ratio

                    if bluest < 10.**(config['color_maxdiff']/2.5):

                        print('set %d: %d arcs, %d images'%(i+1, len(cand.image_sets[i]['arcs']), \
                                                            len(cand.image_sets[i]['images'])))

                        for arc in cand.image_sets[i]['arcs']:
                            print('arc', arc['x'], arc['y'])
                        for image in cand.image_sets[i]['images']:
                            print('image', image['x'], image['y'])

                        foreground_model.update(cand, cand.image_sets[i])

                        #if len(foreground_model.bad_arcs) > 0:
                        #    fitters.fit_bad_arcs(cand, foreground_model, light_model)

                        if len(foreground_model.new_foregrounds) > 0:
                            fitters.fit_new_foregrounds(cand, foreground_model, light_model)

                        fitters.fit_lens(cand, lens_model, light_model, foreground_model, cand.image_sets[i])

                        cand.get_source_footprint()
                        cand.get_footprint_chi2(cand.image_sets[i])

                        cand.get_model_angular_aperture()

                        loglines.append('LENS_MODEL_ANGULAR_APERTURE %2.1f\n'%cand.model_angular_aperture)

                        rchi2 = cand.lensfit_footprint_chi2/(cand.source_footprint.sum())

                        loglines.append('LENS_MODEL_AVG_CHI2 %2.1f'%rchi2)

                        if cand.model_angular_aperture > config['min_aperture']:# and rchi2 < chi2_thresh:

                            success = False

                            if len(arcs) > 1:
                                fitters.fit_ring(cand, ring_model, light_model, foreground_model, cand.image_sets[i])
                                if cand.lensfit_chi2 < cand.ringfit_chi2:
                                    fitters.fit_sersic(cand, sersic_model, light_model, foreground_model, cand.image_sets[i])
                                    if cand.lensfit_chi2 < cand.sersicfit_chi2:
                                        success = True
                                    else:
                                        bestmodel = 'SERSIC'
                                else:
                                    bestmodel = 'RING'

                            else:
                                fitters.fit_sersic(cand, sersic_model, light_model, foreground_model, cand.image_sets[i])
                                if cand.lensfit_chi2 < cand.sersicfit_chi2:
                                    fitters.fit_ring(cand, ring_model, light_model, foreground_model, cand.image_sets[i])
                                    if cand.lensfit_chi2 < cand.ringfit_chi2:
                                        success = True
                                    else:
                                        bestmodel = 'RING'
                                else:
                                    bestmodel = 'SERSIC'

                            if success:
                                isalens = True
                                reason = 'YATTA'
                                print('Yatta!')

                                secondbest = min(cand.sersicfit_chi2, cand.ringfit_chi2)
                                lensness.append((secondbest - cand.lensfit_chi2)/cand.lensfit_chi2)
                                lenssets.append(i)

                                cand.save_model(outname=config['modeldir']+'/%s_model_set%d.fits'%(cand.name, i+1), imset=i, clobber=True)

                                yattargbtools.make_full_rgb(cand, cand.image_sets[i], outname=figname, success=success, config=config)

                            else:
                                if not isalens:
                                    if len(reason) > 0:
                                        reason += '_%s_FITS_BETTER'%bestmodel
                                    else:
                                        reason = '%s_FITS_BETTER'%bestmodel

                                    print('%s fits better'%bestmodel)

                                if config['makeallfigs']:
                                    yattargbtools.make_full_rgb(cand, cand.image_sets[i], outname=figname, success=success, config=config)

                        else:
                            if config['makeallfigs']:
                                yattargbtools.make_full_rgb(cand, cand.image_sets[i], outname=figname, success=False, config=config)

                            if not isalens:
                                if len(reason) > 0:
                                    reason += '_LENSED_ARC_TOO_SMALL'
                                else:
                                    reason = 'LENSED_ARC_TOO_SMALL'
                    else:
                        cand.lensfit_chi2 = None

                        if config['makeallfigs']:
                            yattargbtools.make_full_rgb(cand, cand.image_sets[i], outname=figname, success=None, config=config)

                        if not isalens:
                            if len(reason) > 0:
                                reason += '_ARC_TOO_RED'
                            else:
                                reason = 'ARC_TOO_RED'

                    if config['saveallmodels']:
                        cand.save_model(outname=config['modeldir']+'/%s_model_set%d.fits'%(cand.name, i+1), clobber=True)

                tphase2_end = time.time()
                loglines.append('PHASE_2_TIME %f\n'%(tphase2_end - tphase1_end))

            else:
                if config['makeallfigs']:
                    figname = config['figdir']+'/%s_noarcs.png'%cand.name
                    image_set = {'junk': [obj for obj in objects], 'foregrounds': [], 'arcs': [], 'images': [], \
                                 'bad_arcs': []}
                    yattargbtools.make_full_rgb(cand, image_set=image_set, outname=figname, success=None, config=config)

                if config['saveallmodels']:
                    cand.save_model(outname=config['modeldir']+'/%s_model_set0.fits'%cand.name, clobber=True)

                reason = 'NO_ARCS_FOUND'

        else:
            if config['makeallfigs']:
                figname = config['figdir']+'/%s_nolens.png'%cand.name
                yattargbtools.make_full_rgb(cand, image_set=None, outname=figname, success=None, config=config)

            reason = 'NO_GALAXY_FOUND'

    else:
        data_flag = 1
        reason = 'DATA_NOT_FOUND'

    if isalens:
        zipped = zip(lensness, lenssets)
        slensness, slenssets = zip(*sorted(zipped))

        bset = slenssets[-1]

        summary_line += ' YATTA %d %d\n'%(bset+1, data_flag)

    else:
        summary_line += ' %s 0 %d\n'%(reason, data_flag)

    print('writing output in file %s'%config['summary_file'])

    # cleaning up directory
    if config['cleanupdir']:
        os.system('rm %s/%s_%s_segmap.fits'%(config['modeldir'], cand.name, config['fitband']))
        os.system('rm %s/%s_%s_segmap.fits'%(config['modeldir'], cand.name, config['lightband']))
        os.system('rm %s/%s_%s_secat.cat'%(config['modeldir'], cand.name, config['fitband']))
        os.system('rm %s/%s_%s_secat.cat'%(config['modeldir'], cand.name, config['lightband']))
        os.system('rm %s/%s_%s_lenssub.fits'%(config['modeldir'], cand.name, config['fitband']))
        os.system('rm %s/%s_%s_lenssub.fits'%(config['modeldir'], cand.name, config['lightband']))

    f = open(config['summary_file'], 'a')
    f.writelines([summary_line])
    f.close()

    tend = time.time()
    loglines.append('TOTAL_TIME %f\n'%(tend - tstart))
    logfile = open(config['logdir']+name+'.txt', 'w')
    logfile.writelines(loglines)
    logfile.close()

