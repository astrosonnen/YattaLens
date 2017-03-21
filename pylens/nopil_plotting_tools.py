import numpy as np


def make_rgbarray(images, cuts):

    scaled = []
    for i in range(3):
        img = images[i].copy()

        img[img<0.] = 0.
        img *= 255./cuts[i]
        img[img>255.] = 255.
        img = np.uint8(img.round())
        img = np.flipud(img)
        scaled.append(img.T)

    rgbarray = np.array(scaled).T
    return rgbarray


def make_crazy_pil_format(data, cuts):

    newlist = []
    for i in range(0, 3):
        flatd = np.flipud(data[i]).flatten()
        flatd[flatd<0.] = 0.
        flatd *= 255./cuts[i]
        flatd[flatd>255.] = 255.
        flatd = np.uint8(flatd.round())
        newlist.append(flatd)

    l = []

    for i in range(0, data[0].size):
        l.append((newlist[0][i], newlist[1][i], newlist[2][i]))

    return l


def make_rgb_arrays(candidate, image_set=None, nsig_cut=5.):

    output = []

    cuts = []
    rescuts = []
    data = []
    lenssub = []
    ringresid = []
    lensresid = []
    sersicresid = []
    lensmodel = []
    ringmodel = []
    sersicmodel = []
    source = []

    mask = []
    i = 0

    ncol = 1

    for band in rgbbands:
        img = candidate.sci[band]
        data.append(img)
        cut = np.percentile(img[candidate.R < 30.], 99.)
        cuts.append(cut)
        rescuts.append(np.percentile(img[candidate.R < 30.], 90.))

    s = (data[0].shape[1], data[0].shape[0])

    dlist = make_crazy_pil_format(data, cuts)
    output.append(dlist)

    if len(candidate.lenssub_model) > 0:
        ncol += 1
        for band in rgbbands:
            lenssub.append(candidate.lenssub_resid[band])

        lsublist = make_crazy_pil_format(lenssub, rescuts)
        output.append(lsublist)

        if image_set is not None:
            ncol += 1
            i = 0
            for band in rgbbands:

                maskimg = 0.*candidate.sci[band]

                #for image in image_set['images']:
                #    maskimg[image['footprint'] > 0] = cuts[i]
                for obj in image_set['foregrounds'] + image_set['bad_arcs']:
                    maskimg[obj['footprint'] > 0] = cuts[i]

                if i==0:
                    for junk in image_set['junk']:
                        maskimg[junk['footprint'] > 0] = cuts[i]
                elif i==1:
                    for arc in image_set['arcs']: #+ image_set['foregrounds'] + image_set['bad_arcs']:
                        maskimg[arc['footprint'] > 0] = cuts[i]
                    for image in image_set['images']:
                        maskimg[image['footprint'] > 0] = 0.4 * cuts[i]
                #elif i==2:
                #    for image in image_set['foregrounds'] + image_set['bad_arcs']:
                #        maskimg[image['footprint'] > 0] = cuts[i]

                mask.append(maskimg)
                i += 1

            masklist = make_crazy_pil_format(mask, cuts)
            output.append(masklist)

            x0 = s[1]/2
            y0 = s[0]/2

            if len(candidate.lensfit_model) > 0:
                ncol += 3
                i = 0
                for band in rgbbands:
                    lmodel = 0.*candidate.sci[band]
                    for mimg in candidate.lensfit_model[band]:
                        lmodel += mimg
                    lensmodel.append(lmodel)

                    source.append(candidate.lensfit_model[band][-1])

                    lensresid.append(candidate.sci[band] - lmodel)

                slist = make_crazy_pil_format(source, rescuts)
                lmlist = make_crazy_pil_format(lensmodel, cuts)
                lrlist = make_crazy_pil_format(lensresid, cuts)

                output.append(lmlist)
                output.append(slist)
                output.append(lrlist)

                if len(candidate.ringfit_model) > 0:
                    rcol = ncol
                    ncol += 2
                    for band in rgbbands:
                        rmodel = 0.*img
                        for mimg in candidate.ringfit_model[band]:
                            rmodel += mimg
                        ringmodel.append(rmodel)
                        ringresid.append(candidate.sci[band] - rmodel)

                    rmlist = make_crazy_pil_format(ringmodel, cuts)
                    rrlist = make_crazy_pil_format(ringresid, cuts)

                    output.append(rmlist)
                    output.append(rrlist)

                if len(candidate.sersicfit_model) > 0:
                    scol = ncol
                    ncol += 2
                    for band in rgbbands:
                        smodel = 0.*img
                        for mimg in candidate.sersicfit_model[band]:
                            smodel += mimg
                        sersicmodel.append(smodel)
                        sersicresid.append(candidate.sci[band] - smodel)

                    cmlist = make_crazy_pil_format(sersicmodel, cuts)
                    crlist = make_crazy_pil_format(sersicresid, cuts)

                    output.append(cmlist)
                    output.append(crlist)

    return output

