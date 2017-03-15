import numpy as np
import Image, ImageDraw, ImageFont
from yattaconfig import *



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


def make_fail_curv_rgb(data, lenssub, arcimg, arccoords, arcmask, crapmask=None, fuzzmask=None, maskedge=None, \
                       outname='wrong_curvature.png'):

    cuts = []
    mask = []
    i = 0
    for img in data:
        cut = np.percentile(img, 99.)
        cuts.append(cut)
        if i==0:
            maskdata = arcmask*cut
            if crapmask is not None:
                maskdata += crapmask*cut
            if fuzzmask is not None:
                maskdata += 0.4*fuzzmask*cut
            mask.append(maskdata)
        else:
            mask.append(arcmask*cut)
        i += 1

    dlist = make_crazy_pil_format(data, cuts)
    lsublist = make_crazy_pil_format(lenssub, cuts)
    masklist = make_crazy_pil_format(mask, cuts)
    arclist = make_crazy_pil_format(arcimg, cuts)

    s = data[0].shape
    dim = Image.new('RGB', s, 'black')
    lsubim = Image.new('RGB', s, 'black')
    maskim = Image.new('RGB', s, 'black')
    arcim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    lsubim.putdata(lsublist)
    maskim.putdata(masklist)
    arcim.putdata(arclist)

    x0 = s[1]/2
    y0 = s[0]/2
    if maskedge is not None:
        maskdraw = ImageDraw.Draw(maskim)
        maskdraw.ellipse((x0 - maskedge, y0 - maskedge, x0 + maskedge, y0 + maskedge), fill=None, outline='yellow')

    # draws the lines from the edges of the arc to the center

    def line_y(x, x1, x2, y1, y2):
        return (y2 - y1)/(x2 - x1)*(x - x2) + y2

    def line_x(y, x1, x2, y1, y2):
        return (x2 - x1)/(y2 - y1)*(y - y2) + x2

    def get_vertex(x1, x2, y1, y2):
        if x1 < 0.:
            xedge = 0.
            yedge = line_y(xedge, x1, x2, y1, y2)
            if yedge < 0.:
                yedge = 0.
                xedge = line_x(0., x1, x2, y1, y2)
            elif yedge > s[0]:
                yedge = s[0]
                xedge = line_x(yedge, x1, x2, y1, y2)
        elif x1 > s[1]:
            xedge = s[1]
            yedge = line_y(xedge, x1, x2, y1, y2)
            if yedge < 0.:
                yedge = 0.
                xedge = line_x(yedge, x1, x2, y1, y2)
            elif yedge > s[0]:
                yedge = s[0]
                xedge = line_x(yedge, x1, x2, y1, y2)
        else:
            if y1 < 0.:
                yedge = 0.
                xedge = line_x(yedge, x1, x2, y1, y2)
            elif y1 > s[0]:
                yedge = s[0]
                xedge = line_x(yedge, x1, x2, y1, y2)
            else:
                xedge = x1
                yedge = y1
        return (xedge, yedge)

    draw = ImageDraw.Draw(arcim)

    for coord in arccoords:
        xc, yc = coord[0]
        xa, ya = coord[1]
        xb, yb = coord[2]

        ca_coords = get_vertex(xc, xa, yc, ya)
        cb_coords = get_vertex(xc, xb, yc, yb)


        draw.line((ca_coords[0], s[1] - ca_coords[1], xa, s[1] - ya), fill=(255, 255, 255))
        draw.line((cb_coords[0], s[1] - cb_coords[1], xb, s[1] - yb), fill=(255, 255, 255))

    im = Image.new('RGB', (4*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(lsubim, (s[1], 0))
    im.paste(maskim, (2*s[1], 0))
    im.paste(arcim, (3*s[1], 0))

    im.save(outname)


def make_full_rgb(candidate, image_set=None, maskedge=None, outname='full_model.png', nsig_cut=5., success=None):

    font = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 12)

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
    dim = Image.new('RGB', s, 'black')
    dim.putdata(dlist)

    if len(candidate.lenssub_model) > 0:
        ncol += 1
        for band in rgbbands:
            lenssub.append(candidate.lenssub_resid[band])

        lsublist = make_crazy_pil_format(lenssub, rescuts)
        lsubim = Image.new('RGB', s, 'black')
        lsubim.putdata(lsublist)

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
            maskim = Image.new('RGB', s, 'black')
            maskim.putdata(masklist)

            x0 = s[1]/2
            y0 = s[0]/2
            if maskedge is not None:
                maskdraw = ImageDraw.Draw(maskim)
                maskdraw.ellipse((x0 - maskedge, y0 - maskedge, x0 + maskedge, y0 + maskedge), fill=None, outline='yellow')

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

                sim = Image.new('RGB', s, 'black')
                lmim = Image.new('RGB', s, 'black')
                lrim = Image.new('RGB', s, 'black')

                sim.putdata(slist)
                lmim.putdata(lmlist)
                lrim.putdata(lrlist)

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

                    rmim = Image.new('RGB', s, 'black')
                    rrim = Image.new('RGB', s, 'black')

                    rmim.putdata(rmlist)
                    rrim.putdata(rrlist)

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

                    cmim = Image.new('RGB', s, 'black')
                    crim = Image.new('RGB', s, 'black')

                    cmim.putdata(cmlist)
                    crim.putdata(crlist)

    im = Image.new('RGB', (ncol*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    if ncol > 1:
        im.paste(lsubim, (s[1], 0))
        if ncol > 2:
            im.paste(maskim, (2*s[1], 0))
            if ncol > 5:
                im.paste(lmim, (3*s[1], 0))
                im.paste(sim, (4*s[1], 0))
                im.paste(lrim, (5*s[1], 0))

    draw = ImageDraw.Draw(im)
    draw.text((10, s[0] - 20), 'HSCJ'+candidate.name, font=font, fill='white')
    if ncol > 5:
        draw.text((10 + 5*s[1], s[0] - 20), '%2.1f'%candidate.lensfit_chi2, font=font, fill='white')

    if len(candidate.ringfit_model) > 0:
        im.paste(rmim, (rcol*s[1], 0))
        im.paste(rrim, ((rcol+1)*s[1], 0))
        draw.text((10 + rcol*s[1], s[0] - 20), '%2.1f'%candidate.ringfit_chi2, font=font, fill='white')

    if len(candidate.sersicfit_model) > 0:
        im.paste(cmim, (scol*s[1], 0))
        im.paste(crim, ((scol+1)*s[1], 0))
        draw.text((10 + scol*s[1], s[0] - 20), '%2.1f'%candidate.sersicfit_chi2, font=font, fill='white')

    if success is not None:
        if success:
            draw.ellipse((10, 10, 30, 30), fill=None, outline=(0, 255, 0))
            draw.ellipse((11, 11, 29, 29), fill=None, outline=(0, 255, 0))
        else:
            draw.line((10, 10, 30, 30), fill=(255, 0, 0), width=2)
            draw.line((30, 10, 10, 30), fill=(255, 0, 0), width=2)

    im.save(outname)


def make_model_rgb(sci, light_model, source_model, cuts=(99., 99., 99.), outname='model_rgb.png'):

    auto_cuts = []
    data = []
    lensresid = []
    lensmodel = []

    i = 0

    ncol = 4

    for i in range(3):
        data.append(sci[i])
        cut = np.percentile(sci[i], cuts[i])
        auto_cuts.append(cut)

        lensmodel.append(light_model[i] + source_model[i])

        lensresid.append(sci[i] - light_model[i] - source_model[i])

    dlist = make_crazy_pil_format(data, auto_cuts)
    slist = make_crazy_pil_format(source_model, auto_cuts)

    lmlist = make_crazy_pil_format(lensmodel, auto_cuts)
    lrlist = make_crazy_pil_format(lensresid, auto_cuts)

    s = (data[0].shape[1], data[0].shape[0])
    dim = Image.new('RGB', s, 'black')
    sim = Image.new('RGB', s, 'black')
    lmim = Image.new('RGB', s, 'black')
    lrim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    sim.putdata(slist)
    lmim.putdata(lmlist)
    lrim.putdata(lrlist)

    im = Image.new('RGB', (ncol*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(lmim, (1*s[1], 0))
    im.paste(sim, (2*s[1], 0))
    im.paste(lrim, (3*s[1], 0))

    im.save(outname)


def make_foregroundonly_rgb(candidate, image_set=None, enhance_residuals=False, maskedge=None, \
                            outname='foreground_model.png', nsig_cut=5.):

    font = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 12)

    cuts = []
    rescuts = []
    data = []
    lenssub = []
    lensresid = []
    fgmodel = []
    fgresid = []

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
    dim = Image.new('RGB', s, 'black')
    dim.putdata(dlist)

    if len(candidate.lenssub_model) > 0:
        ncol += 1
        for band in rgbbands:
            lenssub.append(candidate.lenssub_resid[band])

        if enhance_residuals:
            lsublist = make_crazy_pil_format(lenssub, rescuts)
        else:
            lsublist = make_crazy_pil_format(lenssub, cuts)

        lsubim = Image.new('RGB', s, 'black')
        lsubim.putdata(lsublist)

        if image_set is not None:
            ncol += 1
            i = 0
            for band in rgbbands:

                maskimg = 0.*candidate.sci[band]

                for image in image_set['images']:
                    maskimg[image['footprint'] > 0] = cuts[i]

                if i==0:
                    for junk in image_set['junk']:
                        maskimg[junk['footprint'] > 0] = cuts[i]
                elif i==1:
                    for arc in image_set['arcs'] + image_set['foregrounds'] + image_set['bad_arcs']:
                        maskimg[arc['footprint'] > 0] = cuts[i]
                elif i==2:
                    for image in image_set['foregrounds'] + image_set['bad_arcs']:
                        maskimg[image['footprint'] > 0] = cuts[i]

                mask.append(maskimg)
                i += 1

            masklist = make_crazy_pil_format(mask, cuts)
            maskim = Image.new('RGB', s, 'black')
            maskim.putdata(masklist)

            x0 = s[1]/2
            y0 = s[0]/2
            if maskedge is not None:
                maskdraw = ImageDraw.Draw(maskim)
                maskdraw.ellipse((x0 - maskedge, y0 - maskedge, x0 + maskedge, y0 + maskedge), fill=None, outline='yellow')

            if candidate.foreground_model is not None:
                ncol += 2
                i = 0
                for band in rgbbands:
                    lmodel = 0.*candidate.sci[band]
                    for mimg in candidate.foreground_model[band]:
                        lmodel += mimg
                    fgmodel.append(lmodel)

                    fgresid.append(candidate.sci[band] - lmodel)

                fmlist = make_crazy_pil_format(fgmodel, cuts)
                frlist = make_crazy_pil_format(fgresid, cuts)

                fmim = Image.new('RGB', s, 'black')
                frim = Image.new('RGB', s, 'black')

                fmim.putdata(fmlist)
                frim.putdata(frlist)

    im = Image.new('RGB', (ncol*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    if ncol > 1:
        im.paste(lsubim, (s[1], 0))
        if ncol > 2:
            im.paste(maskim, (2*s[1], 0))
            if ncol > 4:
                im.paste(fmim, (3*s[1], 0))
                im.paste(frim, (4*s[1], 0))

    draw = ImageDraw.Draw(im)
    draw.text((10, s[0] - 20), 'HSCJ'+candidate.name, font=font, fill='white')

    im.save(outname)


def make_failure_rgb(candidate, image_set=None, maskedge=None, fail_mode='ring', outname='failed.png', nsig_cut=5., \
                     success=None):

    font = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 12)

    cuts = []
    rescuts = []
    data = []
    lenssub = []
    lensmodel = []
    source = []
    lensresid = []

    altmodel = []
    altonly = []
    altresid = []

    mask = []
    i = 0

    ncol = 3
    nrow = 3

    for band in rgbbands:
        img = candidate.sci[band]
        data.append(img)
        cut = np.percentile(img[candidate.R < 30.], 99.)
        cuts.append(cut)
        rescuts.append(np.percentile(img[candidate.R < 30.], 90.))

        lenssub.append(candidate.lenssub_resid[band])

        maskimg = 0.*candidate.sci[band]

        for obj in image_set['foregrounds'] + image_set['bad_arcs']:
            maskimg[obj['footprint'] > 0] = cuts[i]

        if i==0:
            for junk in image_set['junk']:
                maskimg[junk['footprint'] > 0] = cuts[i]
        elif i==1:
            for arc in image_set['arcs']:
                maskimg[arc['footprint'] > 0] = cuts[i]
            for image in image_set['images']:
                maskimg[image['footprint'] > 0] = 0.4 * cuts[i]

        mask.append(maskimg)

        lmodel = 0.*candidate.sci[band]
        for mimg in candidate.lensfit_model[band]:
            lmodel += mimg
        lensmodel.append(lmodel)

        source.append(candidate.lensfit_model[band][-1])

        lensresid.append(candidate.sci[band] - lmodel)

        rmodel = 0.*img

        if fail_mode == 'ring':
            for mimg in candidate.ringfit_model[band]:
                rmodel += mimg
            altmodel.append(rmodel)
            altonly.append(candidate.ringfit_model[band][-1])
            altresid.append(candidate.sci[band] - rmodel)
        elif fail_mode == 'sersic':
            for mimg in candidate.sersicfit_model[band]:
                rmodel += mimg
            altmodel.append(rmodel)
            altonly.append(candidate.sersicfit_model[band][-1])
            altresid.append(candidate.sci[band] - rmodel)

        i += 1

    slist = make_crazy_pil_format(source, rescuts)
    lmlist = make_crazy_pil_format(lensmodel, cuts)
    lrlist = make_crazy_pil_format(lensresid, rescuts)

    s = (data[0].shape[1], data[0].shape[0])

    lsublist = make_crazy_pil_format(lenssub, rescuts)
    lsubim = Image.new('RGB', s, 'black')
    lsubim.putdata(lsublist)

    dlist = make_crazy_pil_format(data, cuts)
    dim = Image.new('RGB', s, 'black')
    dim.putdata(dlist)

    masklist = make_crazy_pil_format(mask, cuts)
    maskim = Image.new('RGB', s, 'black')
    maskim.putdata(masklist)

    sim = Image.new('RGB', s, 'black')
    lmim = Image.new('RGB', s, 'black')
    lrim = Image.new('RGB', s, 'black')

    sim.putdata(slist)
    lmim.putdata(lmlist)
    lrim.putdata(lrlist)

    amlist = make_crazy_pil_format(altmodel, cuts)
    arlist = make_crazy_pil_format(altresid, rescuts)
    aolist = make_crazy_pil_format(altonly, rescuts)

    amim = Image.new('RGB', s, 'black')
    arim = Image.new('RGB', s, 'black')
    aoim = Image.new('RGB', s, 'black')

    amim.putdata(amlist)
    arim.putdata(arlist)
    aoim.putdata(aolist)

    im = Image.new('RGB', (ncol*data[0].shape[0], nrow*data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(lsubim, (s[1], 0))
    im.paste(maskim, (2*s[1], 0))

    im.paste(lmim, (0, s[0]))
    im.paste(sim, (s[1], s[0]))
    im.paste(lrim, (2*s[1], s[0]))

    im.paste(amim, (0, 2*s[0]))
    im.paste(aoim, (s[1], 2*s[0]))
    im.paste(arim, (2*s[1], 2*s[0]))

    draw = ImageDraw.Draw(im)
    draw.text((10, s[0] - 20), 'HSCJ'+candidate.name, font=font, fill='white')
    draw.text((10 + 2*s[1], 2*s[0] - 20), '%2.1f'%candidate.lensfit_chi2, font=font, fill='white')

    if fail_mode == 'ring':
        draw.text((10 + 2*s[1], 3*s[0] - 20), '%2.1f'%candidate.ringfit_chi2, font=font, fill='white')
    elif fail_mode == 'sersic':
        draw.text((10 + 2*s[1], 3*s[0] - 20), '%2.1f'%candidate.sersicfit_chi2, font=font, fill='white')

    draw.line((10, 10, 30, 30), fill=(255, 0, 0), width=2)
    draw.line((30, 10, 10, 30), fill=(255, 0, 0), width=2)

    im.save(outname)

