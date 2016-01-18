import pylab
import numpy as np
import Image


def make_rgbarray(images, cuts):

    scaled = []
    i = 0
    for img in images:
        img[img<0.] = 0.
        #i99 = np.percentile(img, 99.)
        img *= 255./cuts[i]
        img[img>255.] = 255.
        img = np.uint8(img.round())
        img = np.flipud(img)
        scaled.append(img.T)
        i += 1

    rgbarray = np.array(scaled).T
    return rgbarray


def visual_comparison(data, models, sources=None):

    cuts = []
    resids = []
    i = 0
    for img in data:
        cuts.append(np.percentile(img, 99.))
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


def make_rgb_png(data, models, sources=None, outname='rgb.png'):

    cuts = []
    resids = []
    i = 0
    for img in data:
        cuts.append(np.percentile(img, 99.))
        resids.append(img - models[i])
        i += 1

    dlist = make_crazy_pil_format(data, cuts)
    mlist = make_crazy_pil_format(models, cuts)
    rlist = make_crazy_pil_format(resids, cuts)


    s = data[0].shape
    dim = Image.new('RGB', s, 'black')
    mim = Image.new('RGB', s, 'black')
    rim = Image.new('RGB', s, 'black')

    dim.putdata(dlist)
    mim.putdata(mlist)
    rim.putdata(rlist)

    if sources is None:
        im = Image.new('RGB', (3*data[0].shape[0], data[0].shape[1]), 'black')
        rindex = 2
    else:
        im = Image.new('RGB', (4*data[0].shape[0], data[0].shape[1]), 'black')
        sim = Image.new('RGB', s, 'black')
        slist = make_crazy_pil_format(sources, cuts)
        sim.putdata(slist)
        im.paste(sim, (2*s[1], 0))
        rindex = 3

    im.paste(dim, (0, 0,))
    im.paste(mim, (s[1], 0))
    im.paste(rim, (rindex*s[1], 0))

    im.save(outname)
