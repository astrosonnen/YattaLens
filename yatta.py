from yattaconfig import *
import numpy as np
import sys
import yattaobjects as yo
from pylens import plotting_tools
import object_finder_tools as oft
import pickle
import fitters

name = sys.argv[1]

cand = yo.Candidate(name=name, bands=('i', 'r', 'g'))

if cand.read_data():

    light_model = yo.light_model(cand)

    fitters.quick_lens_subtraction(cand, light_model, rmax=20., lfitband=('i'), niter=200)

    objects, segmap, foundarcs, arclist = oft.find_objects(cand, detect_band='g', detect_thresh=3.)

    if foundarcs:

        iobjs, crapmask, foundarcs, arclist = oft.find_objects(cand, detect_band='i', detect_thresh=2.)
        crapmask[crapmask > 0] = 1
        crapmask = 1 - crapmask
        crapmask[cand.R < 5.] = 1

        fitters.fit_light(cand, light_model, lfitband=('i'), mask=crapmask, nsamp=200, rmax=20.)

        objects, segmap, foundarcs, arclist = oft.find_objects(cand, detect_band='g', detect_thresh=3., meas_bands=['i', 'g'])

        if foundarcs:

            cand.image_sets = oft.determine_image_sets(objects, arclist)

            nsets = len(cand.image_sets)

            lens_model = yo.lens_model(cand)
            ring_model = yo.ring_model(cand)

            for i in range(nsets):

                fitters.fit_lens(cand, lens_model, light_model, cand.image_sets[i])

                fitters.fit_ring(cand, ring_model, light_model, image_set=cand.image_sets[i])

                cand.get_source_footprint()
                cand.get_footprint_chi2(cand.image_sets[i])

                print cand.lensfit_footprint_chi2, cand.ringfit_footprint_chi2

                plotting_tools.make_full_rgb(cand, cand.image_sets[i], outname='figs/%s_imset%d.png'%(cand.name, i+1))

		f = open(modeldir+'/%s_model_set%d.dat'%(cand.name, i+1), 'w')
		pickle.dump(cand, f)
		f.close()



