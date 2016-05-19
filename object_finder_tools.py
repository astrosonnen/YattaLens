from yattaconfig import *
import pyfits
import numpy as np
from scipy.optimize import minimize
import os


def find_objects(candidate, detect_band='g', detect_thresh=3.):

    lsubname = modeldir+'/%s_%s_lenssub.fits'%(candidate.name, detect_band)
    varname = datadir+'/%s_%s_var.fits'%(candidate.name, detect_band)

    segname = modeldir+'/%s_%s_segmap.fits'%(candidate.name, detect_band)
    catname = modeldir+'/%s_%s_secat.cat'%(candidate.name, detect_band)

    pyfits.PrimaryHDU(candidate.lenssub_resid[detect_band]).writeto(lsubname, clobber=True)

    os.system('sex %s -c seconfig.sex -WEIGHT_IMAGE %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s -DETECT_THRESH %f'%\
              (lsubname, varname, catname, segname, detect_thresh))

    f = open(catname, 'r')
    cat = np.atleast_2d(np.loadtxt(f))
    f.close()

    cs = cat.shape
    if cs[1] == 0:
        nobj = 0

    else:
        nobj = cs[0]

    segmap = pyfits.open(segname)[0].data.copy()

    objects = {}
    foundarcs = False
    arclist = []

    for i in range(nobj):
        ind = i+1
        obj = {}
        x = cat[i, 0] - 1
        y = cat[i, 1] - 1
        obj['x'] = x
        obj['y'] = y
        obj['r'] = ((x - candidate.x0) ** 2 + (y - candidate.y0) ** 2) ** 0.5
        theta = np.rad2deg(np.arctan(-(x - candidate.x0) / (y - candidate.y0)))
        obj['ang_diff'] = min(abs(theta - cat[i, 5]), abs(abs(theta - cat[i, 5]) - 180.))
        obj['ab'] = cat[i, 4]
        obj['npix'] = cat[i, 3]

        footprint = np.zeros(segmap.shape, dtype=int)
        footprint[segmap==ind] = 1

        obj['footprint'] = footprint

        if obj['r'] > minarcdist and obj['r'] < maxarcdist and obj['npix'] < maxarcsize \
            and obj['npix'] > minarcsize and obj['ang_diff'] < maxarcdang and obj['ab'] > abmin:
            foundarcs = True
            arclist.append(i)
            obj['arclike'] = True
        else:
            obj['arclike'] = False

        objects[i] = obj

    return objects, segmap, foundarcs, arclist


def measure_fluxes(objects, candidate, meas_bands=('g', 'i'), model_err=0.1):

    nobj = len(objects)
    for i in objects:
        obj = objects[i]
        for band in meas_bands:
            obj['%s_flux'%band] = candidate.lenssub_resid[band][obj['footprint'] > 0].sum()
            modeling_err = model_err*candidate.lenssub_model[band][obj['footprint'] > 0].sum()
            obj['%s_err'%band] = (candidate.var[band][obj['footprint'] > 0].sum() + modeling_err**2)**0.5

    return objects


def compare_colors(f1a, f1b, f2a, f2b, e1a, e1b, e2a, e2b, nsigma=4.):
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

    return samecolor, res.x, ml


def color_compatibility(objects, band1='g', band2='i', nsigma=4.):

    nobj = len(objects)
    # checks color compatibility
    color_matrix = np.ones((nobj, nobj), dtype=int)

    for i in range(nobj):
        obj1 = objects[i]
        for j in range(i+1, nobj):
            obj2 = objects[j]

            samecolor, pars, logp = compare_colors(obj1['%s_flux'%band1], obj1['%s_flux'%band2], \
                                                   obj2['%s_flux'%band1], obj2['%s_flux'%band2], \
                                                   obj1['%s_err'%band1], obj1['%s_err'%band2], \
                                                   obj2['%s_err'%band1], obj2['%s_err'%band2], nsigma=nsigma)

            if not samecolor:
                color_matrix[i, j] = 0
                color_matrix[j, i] = 0

    return color_matrix


def determine_image_sets(objects, arcs):

    color_matrix = color_compatibility(objects)

    narcs = len(arcs)
    nobj = len(objects)

    image_sets = []
    done_combinations = []

    for i in range(narcs):

        arc = arcs[i]

        arcs_here = []
        image_set = {'arcs': [objects[arc]], 'images': [], 'junk': [], 'furthest_arc': None, 'mean_arc_dist': None}

        arcs_here.append(arc)

        furthest = objects[arc]['r']
        meandist = objects[arc]['r']

        colorcheck = color_matrix[arc]

        for j in range(narcs):
            if j != i:
                if color_matrix[arcs[i], arcs[j]] == 1:
                    arcs_here.append(arcs[j])
                    image_set['arcs'].append(objects[arcs[j]])
                    colorcheck *= color_matrix[arcs[j]]
                    if objects[arcs[j]]['r'] > furthest:
                        furthest = objects[arcs[j]]['r']
                    meandist += objects[arcs[j]]['r']

                else:
                    image_set['junk'].append(objects[arcs[j]])

        meandist /= float(len(image_set['arcs']))

        for j in range(nobj):
            if j not in arcs_here:
                if colorcheck[j] == 1 and objects[j]['r'] < crapstart*furthest:
                    image_set['images'].append(objects[j])
                else:
                    image_set['junk'].append(objects[j])

        image_set['furthest_arc'] = furthest
        image_set['mean_arc_dist'] = meandist

        arcs_here.sort()
        if arcs_here not in done_combinations:
            done_combinations.append(arcs_here)

            image_sets.append(image_set)

    return image_sets

