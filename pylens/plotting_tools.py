import pylab
import numpy as np
import Image
import ImageDraw


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


def visual_comparison(data, models, sources=None):

    cuts = []
    resids = []
    i = 0
    for img in data:
        cut = np.percentile(img, 99.)
        cuts.append(cut)
        resids.append(img - models[i])
        i += 1

    #pylab.subplots_adjust(left=0., right=1., bottom=0., top=1., hspace=0., wspace=0.)
    pylab.subplot(2, 2, 1)
    pylab.imshow(make_rgbarray(data, cuts))
    pylab.xticks(())
    pylab.yticks(())

    pylab.subplot(2, 2, 2)
    pylab.imshow(make_rgbarray(models, cuts))
    pylab.xticks(())
    pylab.yticks(())

    pylab.subplot(2, 2, 3)
    pylab.imshow(make_rgbarray(resids, cuts))
    pylab.xticks(())
    pylab.yticks(())

    if sources is not None:
        pylab.subplot(2, 2, 4)
        pylab.imshow(make_rgbarray(sources, cuts))
        pylab.xticks(())
        pylab.yticks(())


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


def make_rgb_png(data, lenssub, model, source, arcmask, crapmask=None, maskedge=None, outname='rgb.png'):

    cuts = []
    resid = []
    mask = []
    i = 0
    for img in data:
        cut = np.percentile(img, 99.)
        cuts.append(cut)
        resid.append(img - model[i])
        if i==0:
            if crapmask is not None:
                mask.append(arcmask*cut + crapmask*cut)
        else:
            mask.append(arcmask*cut)
        i += 1

    dlist = make_crazy_pil_format(data, cuts)
    lsublist = make_crazy_pil_format(lenssub, cuts)
    mlist = make_crazy_pil_format(model, cuts)
    rlist = make_crazy_pil_format(resid, cuts)
    alist = make_crazy_pil_format(mask, cuts)
    slist = make_crazy_pil_format(source, cuts)

    s = data[0].shape
    dim = Image.new('RGB', s, 'black')
    lsubim = Image.new('RGB', s, 'black')
    aim = Image.new('RGB', s, 'black')
    mim = Image.new('RGB', s, 'black')
    sim = Image.new('RGB', s, 'black')
    rim = Image.new('RGB', s, 'black')

    if maskedge is not None:
        x0 = s[1]/2
        y0 = s[0]/2
        maskdraw = ImageDraw.Draw(aim)
        maskdraw.ellipse((x0 - maskedge, y0 - maskedge, x0 + maskedge, y0 + maskedge), fill=None, outline='yellow')

    dim.putdata(dlist)
    lsubim.putdata(lsublist)
    mim.putdata(mlist)
    rim.putdata(rlist)
    aim.putdata(alist)
    sim.putdata(slist)

    im = Image.new('RGB', (6*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(lsubim, (s[1], 0))
    im.paste(aim, (2*s[1], 0))
    im.paste(mim, (3*s[1], 0))
    im.paste(sim, (4*s[1], 0))
    im.paste(rim, (5*s[1], 0))

    im.save(outname)

def make_long_rgb_png(data, lenssub, arcimg, model, source, arcmask, normresid, nsig_cut=5., crapmask=None, \
                      maskedge=None, fuzzmask=None, outname='rgb.png'):

    cuts = []
    resid = []
    lensonly_resid = []
    mask = []
    normresid_rgb = []
    normresid_cuts = []
    i = 0
    for img in data:
        cut = np.percentile(img, 99.)
        cuts.append(cut)
        resid.append(img - model[i])
        lensonly_resid.append(img - model[i] + source[i])
        normresid_rgb.append(normresid + nsig_cut)
        normresid_cuts.append(2.*nsig_cut)
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
    lensresidlist = make_crazy_pil_format(lensonly_resid, cuts)
    mlist = make_crazy_pil_format(model, cuts)
    slist = make_crazy_pil_format(source, cuts)
    rlist = make_crazy_pil_format(resid, cuts)
    normresidlist = make_crazy_pil_format(normresid_rgb, normresid_cuts)

    s = data[0].shape
    dim = Image.new('RGB', s, 'black')
    lsubim = Image.new('RGB', s, 'black')
    maskim = Image.new('RGB', s, 'black')
    arcim = Image.new('RGB', s, 'black')
    lresim = Image.new('RGB', s, 'black')
    mim = Image.new('RGB', s, 'black')
    sim = Image.new('RGB', s, 'black')
    rim = Image.new('RGB', s, 'black')
    nim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    lsubim.putdata(lsublist)
    maskim.putdata(masklist)
    arcim.putdata(arclist)
    lresim.putdata(lensresidlist)
    mim.putdata(mlist)
    sim.putdata(slist)
    rim.putdata(rlist)
    nim.putdata(normresidlist)

    x0 = s[1]/2
    y0 = s[0]/2
    if maskedge is not None:
        maskdraw = ImageDraw.Draw(maskim)
        maskdraw.ellipse((x0 - maskedge, y0 - maskedge, x0 + maskedge, y0 + maskedge), fill=None, outline='yellow')

    im = Image.new('RGB', (8*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(lsubim, (s[1], 0))
    im.paste(maskim, (2*s[1], 0))
    #im.paste(arcim, (3*s[1], 0))
    im.paste(lresim, (3*s[1], 0))
    im.paste(mim, (4*s[1], 0))
    im.paste(sim, (5*s[1], 0))
    im.paste(rim, (6*s[1], 0))
    im.paste(nim, (7*s[1], 0))

    im.save(outname)


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


def make_fuzz_rgb(data, lenssub, arcmask=None, crapmask=None, fuzzmask=None, maskedge=None, \
                  outname='too_much_fuzz.png'):

    cuts = []
    mask = []
    i = 0
    for img in data:
        cut = np.percentile(img, 99.)
        cuts.append(cut)
        if i==0:
            mask.append(arcmask*cut + crapmask*cut + 0.4*fuzzmask*cut)
        else:
            mask.append(arcmask*cut)
        i += 1

    dlist = make_crazy_pil_format(data, cuts)
    lsublist = make_crazy_pil_format(lenssub, cuts)
    masklist = make_crazy_pil_format(mask, cuts)

    s = data[0].shape
    dim = Image.new('RGB', s, 'black')
    lsubim = Image.new('RGB', s, 'black')
    maskim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    lsubim.putdata(lsublist)
    maskim.putdata(masklist)

    x0 = s[1]/2
    y0 = s[0]/2
    if maskedge is not None:
        maskdraw = ImageDraw.Draw(maskim)
        maskdraw.ellipse((x0 - maskedge, y0 - maskedge, x0 + maskedge, y0 + maskedge), fill=None, outline='yellow')

    im = Image.new('RGB', (3*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(lsubim, (s[1], 0))
    im.paste(maskim, (2*s[1], 0))

    im.save(outname)


def make_arcfinder_mask_png(data, mask, outname='arcfinder_mask.png'):

    cuts = []
    i = 0
    mlist = []
    for img in data:
        cut = np.percentile(img, 99.)
        cuts.append(cut)
        mlist.append(mask*cut)
        i += 1

    dlist = make_crazy_pil_format(data, cuts)
    mlist = make_crazy_pil_format(mlist, cuts)


    s = data[0].shape
    dim = Image.new('RGB', s, 'black')
    mim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    mim.putdata(mlist)

    im = Image.new('RGB', (2*data[0].shape[0], data[0].shape[1]), 'black')

    im.paste(dim, (0, 0,))
    im.paste(mim, (s[1], 0))

    im.save(outname)
