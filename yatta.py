from yattaconfig import *
import numpy as np
import sys
import yattaobjects as yo

name = sys.argv[1]

cand = yo.Candidate(name=name, bands=['i', 'r', 'g'])

if cand.read_data():

    cand.prepare_model()

    cand.quick_lens_subtraction(rmax=20., fitband=['i'], niter=200)

    objects, segmap, yesarcs = cand.find_objects(detect_band='g', detect_thresh=3.)

    if yesarcs:

        iobjs, crapmask = cand.find_objects(detect_band='i', detect_thresh=2.)
        crapmask[cand.R < 5.] = 0

        cand.lens_subtraction(fitband=['i'], mask=crapmask)

        objects, segmap, yesarcs = cand.find_objects(detect_band='g', detect_thresh=3., meas_bands=['i', 'g'])

        if yesarcs:
            for ind in objects:
                if objects[ind]['arclike']:
                    print objects[ind]

