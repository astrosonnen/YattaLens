from yattaconfig import *
import pyfits
import numpy as np
from scipy.optimize import minimize
import os


def find_lens(candidate, detect_band='i', detect_thresh=3.):

    sciname = datadir+'/%s_%s_sci.fits'%(candidate.name, detect_band)
    varname = datadir+'/%s_%s_var.fits'%(candidate.name, detect_band)

    segname = modeldir+'/%s_%s_segmap.fits'%(candidate.name, detect_band)
    catname = modeldir+'/%s_%s_secat.cat'%(candidate.name, detect_band)

    os.system('sex %s -c $YATTADIR/sedir/seconfig.sex -WEIGHT_IMAGE %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -DETECT_THRESH %f'%\
              (sciname, varname, catname, segname, detect_thresh))

    f = open(catname, 'r')
    cat = np.atleast_2d(np.loadtxt(f))
    f.close()

    cs = cat.shape
    if cs[1] == 0:
        nobj = 0
    else:
        nobj = cs[0]

    if nobj > 0:
        segmap = pyfits.open(segname)[0].data.copy()
    else:
        segmap = np.zeros(candidate.imshape, dtype=int)

    junkmask = np.ones(candidate.imshape, dtype='int')

    objects = []

    closestdist = np.inf
    closestind = -1

    for i in range(nobj):
        ind = i+1
        obj = {}
        x = cat[i, 0] - 1.
        y = cat[i, 1] - 1.
        obj['x'] = x
        obj['y'] = y
        obj['r'] = ((x - candidate.x0) ** 2 + (y - candidate.y0) ** 2) ** 0.5
        obj['pa'] = cat[i, 5]
        obj['ab'] = cat[i, 4]
        obj['npix'] = cat[i, 3]
        if obj['r'] < closestdist and obj['npix'] > 50:
            closestdist = obj['r']
            closestind = ind

        footprint = np.zeros(segmap.shape, dtype=int)
        footprint[segmap==ind] = 1

        obj['footprint'] = footprint

        objects.append(obj)

    lens = None
    for i in range(nobj):
        ind = i+1
        if ind == closestind:
            lens = objects[i]
        else:
            junkmask[objects[i]['footprint'] > 0] = 0

    return lens, junkmask


def find_objects(candidate, detect_band='g', detect_thresh=3.):

    lsubname = modeldir+'/%s_%s_lenssub.fits'%(candidate.name, detect_band)
    varname = datadir+'/%s_%s_var.fits'%(candidate.name, detect_band)

    segname = modeldir+'/%s_%s_segmap.fits'%(candidate.name, detect_band)
    catname = modeldir+'/%s_%s_secat.cat'%(candidate.name, detect_band)

    pyfits.PrimaryHDU(candidate.lenssub_resid[detect_band]).writeto(lsubname, clobber=True)

    os.system('sex %s -c $YATTADIR/sedir/seconfig.sex -WEIGHT_IMAGE %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -DETECT_THRESH %f'%\
              (lsubname, varname, catname, segname, detect_thresh))

    f = open(catname, 'r')
    cat = np.atleast_2d(np.loadtxt(f))
    f.close()

    cs = cat.shape
    if cs[1] == 0:
        nobj = 0

    else:
        nobj = cs[0]

    if nobj > 0:
        segmap = pyfits.open(segname)[0].data.copy()
    else:
        segmap = np.zeros(candidate.imshape, dtype=int)

    objects = []
    foundarcs = False
    arcs = []

    for i in range(nobj):
        ind = i+1
        obj = {}
        x = cat[i, 0] - 1
        y = cat[i, 1] - 1
        obj['x'] = x
        obj['y'] = y
        obj['r'] = ((x - candidate.x0) ** 2 + (y - candidate.y0) ** 2) ** 0.5
        obj['pa'] = cat[i, 5]
        theta = np.rad2deg(np.arctan(-(x - candidate.x0) / (y - candidate.y0)))
        obj['ang_diff'] = min(abs(theta - cat[i, 5]), abs(abs(theta - cat[i, 5]) - 180.))
        obj['ab'] = cat[i, 4]
        obj['npix'] = cat[i, 3]

        # obj['angap'] = 2.*(obj['npix']/np.pi*obj['ab'])**0.5/obj['r']*180./np.pi

        footprint = np.zeros(segmap.shape, dtype=int)
        footprint[segmap==ind] = 1

        obj['footprint'] = footprint

        xpix = candidate.X[footprint > 0]
        ypix = candidate.Y[footprint > 0]

        rpix = ((xpix - candidate.x)**2 + (ypix - candidate.y)**2)**0.5
        cospix = (xpix - candidate.x)/rpix
        sinpix = (ypix - candidate.y)/rpix

        npix = len(xpix)

        max_aperture = 0.
        for j in range(npix):
            cosdiff = cospix[j]*cospix + sinpix[j]*sinpix
            aperture = 180.*np.arccos(cosdiff).max()/np.pi
            if aperture > max_aperture:
                max_aperture = aperture

        obj['angap'] = max_aperture

        if obj['r'] > minarcdist and obj['r'] < maxarcdist and obj['npix'] < maxarcsize and obj['npix'] > minarcsize \
                and obj['ang_diff'] < maxarcdang and obj['ab'] > abmin and obj['angap'] > se_minap:
            foundarcs = True
            arcs.append(obj)
        elif obj['r'] > minobjdist:
            objects.append(obj)

    return objects, arcs, segmap, foundarcs


def measure_fluxes(objects, candidate, foreground_model, meas_bands=('g', 'i'), model_err=0.1):

    for obj in objects:
        for band in meas_bands:
            model = candidate.foreground_model[band][0].copy()
            ncomp = 1
            for comp in foreground_model.components:
                xcomp = comp['pars'][0].value
                ycomp = comp['pars'][1].value

                if int(round(ycomp)) >= 0 and int(round(ycomp)) < candidate.imshape[0] and int(round(xcomp)) >= 0 and int(round(xcomp)) < candidate.imshape[1]:
                    if obj['footprint'][int(round(ycomp)), int(round(xcomp))] == 0:
                        model += candidate.foreground_model[band][ncomp]
                ncomp += 1

            resid = candidate.sci[band] - model

            obj['%s_flux'%band] = resid[obj['footprint'] > 0].sum()
            modeling_err = model_err*model[obj['footprint'] > 0].sum()

            obj['%s_err'%band] = (candidate.var[band][obj['footprint'] > 0].sum() + modeling_err**2)**0.5

    return objects


def compare_colors(f1a, f1b, f2a, f2b, e1a, e1b, e2a, e2b, nsigma=color_nsigma):
    """
    Compares fluxes of two objects, 1 and 2, in two bands, a and b, and determines whether they have the same color
    :param f1a: band a flux of object 1
    :param f1b: band b flux of object 1
    :param f2a: band a flux of object 2
    :param f2b: band b flux of object 2
    :param e1a: measurement error on f1a
    :param e1b: measurement error on f1b
    :param e2a: measurement error on f2a
    :param e2b: measurement error on f2b
    :param nsigma: number of sigmas required to reject the null hypothesis of identical color
    :return: consistent: bool-type
    """

    """
    def nloglike(p):
        f1, f2, c = p
        t1a = -0.5*(f1a - f1)**2/e1a**2
        t1b = -0.5*(f1b - f1*c)**2/e1b**2
        t2a = -0.5*(f2a - f2)**2/e2a**2
        t2b = -0.5*(f2b - f2*c)**2/e2b**2
        return -t1a -t1b -t2a -t2b

    guess = (max(f1a, e1a), max(f2a, e2a), max(f1a/f1b, 0.01))

    res = minimize(nloglike, guess, method='L-BFGS-B', bounds=((0., f1a + 5.*e1a), (0., f2a + 5.*e2a), \
                                                               (0.001, 1000.)), tol=0.1)

    ml = -nloglike(res.x)

    if ml < -0.5*nsigma:
        samecolor = False
    else:
        samecolor = True
    """

    r1 = f1b/f1a
    r1err = r1*(e1b/f1b + e1a/f1a)

    r2 = f2b/f2a
    r2err = r2*(e2b/f2b + e2a/f2a)

    ratios = [r1, r2]
    errors = [r1err, r2err]

    tmp = zip(ratios, errors)
    tmp.sort()

    ratios, errors = zip(*tmp)

    if ratios[0] + nsigma*errors[0] > ratios[1] - nsigma*errors[1]:
        samecolor = True
    else:
        samecolor = False

    return samecolor


def color_compatibility(objects, band1='g', band2='i', nsigma=color_nsigma):

    nobj = len(objects)
    # checks color compatibility
    color_matrix = np.ones((nobj, nobj), dtype=int)

    for i in range(nobj):
        obj1 = objects[i]
        for j in range(i+1, nobj):
            obj2 = objects[j]

            samecolor = compare_colors(obj1['%s_flux'%band1], obj1['%s_flux'%band2], \
                                                   obj2['%s_flux'%band1], obj2['%s_flux'%band2], \
                                                   obj1['%s_err'%band1], obj1['%s_err'%band2], \
                                                   obj2['%s_err'%band1], obj2['%s_err'%band2], nsigma=nsigma)

            if not samecolor:
                color_matrix[i, j] = 0
                color_matrix[j, i] = 0

    return color_matrix


def determine_image_sets(objects, arcs, iobjects, band1='g', band2='i'):

    color_matrix = color_compatibility(objects + arcs, band1=band1, band2=band2)

    narcs = len(arcs)
    nobj = len(objects)

    niobj = len(iobjects)

    image_sets = []
    done_combinations = []

    for i in range(narcs):

        arc = arcs[i]

        arcs_here = []
        image_set = {'arcs': [arc], 'bad_arcs': [], 'images': [], 'foregrounds': [], 'junk': [], 'furthest_arc': None, \
                     'mean_arc_dist': None}

        arcs_here.append(i)

        furthest = arc['r']
        meandist = arc['r']
        brightest = arc['%s_flux'%band1]

        colorcheck = color_matrix[nobj + i]

        tmp_bad_arcs = []
        for j in range(narcs):
            if j != i:
                if color_matrix[nobj + i, nobj + j] == 1:
                    arcs_here.append(j)
                    image_set['arcs'].append(arcs[j])
                    colorcheck *= color_matrix[nobj + j]
                    if arcs[j]['r'] > furthest:
                        furthest = arcs[j]['r']
                    if arcs[j]['%s_flux'%band1] > brightest:
                        brightest = arcs[j]['%s_flux'%band1]
                    meandist += arcs[j]['r']

                else:
                    tmp_bad_arcs.append(arcs[j])
                    #image_set['bad_arcs'].append(arcs[j])

        for bad_arc in tmp_bad_arcs:
            if bad_arc['r'] < modeluntil*furthest:
                #image_set['bad_arcs'].append(bad_arc)
                image_set['foregrounds'].append(bad_arc)
            else:
                image_set['junk'].append(bad_arc)

        meandist /= float(len(image_set['arcs']))

        for j in range(nobj):
            if objects[j]['r'] > junkstart*furthest:
                image_set['junk'].append(objects[j])
            elif colorcheck[j] == 1 and objects[j]['%s_flux'%band1] < 2.*brightest:
                image_set['images'].append(objects[j])
            else:
                image_set['foregrounds'].append(objects[j])

        for j in range(niobj):

            xiobj = iobjects[j]['x']
            yiobj = iobjects[j]['y']
            already_there = False
            for n in range(nobj):
                if objects[n]['footprint'][int(round(yiobj)), int(round(xiobj))] > 0:
                    already_there = True
            for n in range(narcs):
                if arcs[n]['footprint'][int(round(yiobj)), int(round(xiobj))] > 0:
                    already_there = True
            if not already_there:
                print 'adding footprint of object at %2.1f %2.1f. dist: %2.1f. arc dist: %2.1f'%(xiobj, yiobj, iobjects[j]['r'], furthest)
                if iobjects[j]['r'] <= junkstart*furthest:
                    image_set['foregrounds'].append(iobjects[j])
                else:
                    image_set['junk'].append(iobjects[j])

        image_set['furthest_arc'] = furthest
        image_set['mean_arc_dist'] = meandist

        arcs_here.sort()
        if arcs_here not in done_combinations:
            done_combinations.append(arcs_here)

            image_sets.append(image_set)

    return image_sets

