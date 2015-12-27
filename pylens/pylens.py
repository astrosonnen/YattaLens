import numpy as np
from scipy import optimize
from scipy.optimize import fmin_slsqp
from photofit import convolve

class LensModel:
    """
    IN PROGRESS
    The purpose of this class is to allow lens modelling with e.g., caching to
        update parts of the model independently.
    """
    def __init__(self,img,sig,gals,lenses,srcs,psf=None):
        self.img = img
        self.sig = sig
        self.gals = gals
        self.lenses = lenses
        self.srcs = srcs
        self.psf = psf

        self.rhs = (img/sig).flatten()
        self.fsig = sig.flatten()
        self.fit

        self.mask = None

        self.logp = None
        self.model = None
        self.amps = None
        self.components = None

    def addMask(self,mask):
        self.rhs = (img/sig)[mask].flatten()
        self.fsig = sig[mask].flatten()

    def fit(x,y,fast=False):
        mask = self.mask
        gals = self.gals
        srcs = self.srcs
        lenses = self.lenses
        rhs = self.rhs
        sig = self.fsig

        if mask is not None:
            xin = x[mask].copy()
            yin = y[mask].copy()
        else:
            xin = x.copy()
            yin = y.copy()

        model = numpy.empty((len(gals)+len(srcs),rhs.size))
        n = 0
        for gal in gals:
            gal.setPars()
            gal.amp = 1.
            tmp = gal.pixeval(xin,yin,1./self.oversample,csub=self.csub)
            if numpy.isnan(tmp):
                self.model = None
                self.amps = None
                self.components = None
                self.logp = -1e300
                return -1e300
            if psf is not None and gal.convolve is not None:
                if mask is None:
                    model[n] = convolve.convolve(tmp,psf,False)[0]
                else:
                    model[n] = 1 
    
        gals = self.gals


def getDeflections(massmodels,points):
    if type(points)==type([]) or type(points)==type(()):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    if type(massmodels)!=type([]):
        massmodels = [massmodels]
    x0 = x.copy()
    y0 = y.copy()
    for massmodel in massmodels:
        xmap,ymap = massmodel.deflections(x,y)
        y0 -= ymap
        x0 -= xmap
    return x0.reshape(x.shape),y0.reshape(y.shape)



def lens_images(massmodels,sources,points,factor=1,getPix=False):
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    if type(massmodels)!=type([]):
        massmodels = [massmodels]
#    x0 = x.flatten()
#    y0 = y.flatten()
    x0 = x.copy()
    y0 = y.copy()
    for massmodel in massmodels:
        xmap,ymap = massmodel.deflections(x,y)
        y0 -= ymap#.reshape(y.shape)#/scale
        x0 -= xmap#.reshape(x.shape)#/scale
    x0,y0 = x0.reshape(x.shape),y0.reshape(y.shape)
    if getPix==True:
        return x0,y0
    if type(sources)!=type([]):
        sources = [sources]
    out = x*0.
    for src in sources:
        out += src.pixeval(x0,y0,factor,csub=11)
    return out


def dblPlane(scales,massmodels,sources,points,factor):
    if type(points)==type([]):
        x1,y1 = points[0].copy(),points[1].copy()
    else:
        y1,x1 = points[0].copy(),points[1].copy()
    out = x1*0.
    ax_1 = x1*0.
    ay_1 = x1*0.
    for l in massmodels[0]:
        xmap,ymap = l.deflections(x1,y1)
        ax_1 += xmap.reshape(ax_1.shape)
        ay_1 += ymap.reshape(ay_1.shape)
    for s in sources[0]:
        out += s.pixeval(x1-ax_1,y1-ay_1,factor,csub=11)
    x2 = x1-scales[0,0]*ax_1
    y2 = y1-scales[0,0]*ay_1
    ax_2 = x2*0.
    ay_2 = y2*0.
    for l in massmodels[1]:
        xmap,ymap = l.deflections(x2,y2)
        ax_2 += xmap.reshape(ax_2.shape)
        ay_2 += ymap.reshape(ay_2.shape)
    for s in sources[1]:
        out += s.pixeval(x1-ax_1-ax_2,y1-ay_1-ay_2,factor,csub=11)
    return out


def multiplePlanes(scales,massmodels,points):
    from numpy import zeros,eye,triu_indices
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    out = x*0.
    nplanes = len(massmodels)
    tmp = scales.copy()
    scales = eye(nplanes)
    scales[triu_indices(nplanes,1)] = tmp
    ax = zeros((x.shape[0],x.shape[1],nplanes))
    ay = ax.copy()
    x0 = x.copy()
    y0 = y.copy()
    out = []
    for p in range(nplanes):
        for massmodel in massmodels[p]:
            xmap,ymap = massmodel.deflections(x0,y0)
            ax[:,:,p] += xmap.reshape(x.shape)
            ay[:,:,p] += ymap.reshape(y.shape)
        x0 = x-(ax[:,:,:p+1]*scales[:p+1,p]).sum(2)
        y0 = y-(ay[:,:,:p+1]*scales[:p+1,p]).sum(2)
        out.append([x0,y0])
    return out


def objf(x, lhs, rhs):
    return ((np.dot(lhs, x) - rhs)**3).sum()
def objdf(x, lhs, rhs):
    return np.dot(lhs.T, np.dot(lhs, x) - rhs)


def getModel_sourceonly(lens, source, spars, image, sigma, mask, X, Y, zp=30., returnImg=False):

    source.setPars(spars)
    lens.setPars()

    simage = ((image/sigma).ravel())[mask.ravel()]

    xl, yl = getDeflections([lens], [X, Y])
    simg = source.pixeval(xl, yl)
    simg = convolve.convolve(simg, source.convolve, False)[0]

    if np.isnan(simg).any():
        rimg = 0.*image
        rimg[simg!=simg] = 1.
        return 0., 0., rimg

    nmod = 1

    model = np.zeros((nmod, mask.sum()))
    norm = np.zeros(nmod)

    model[0] = simg[mask].ravel()
    norm[0] = model[0].max()
    model[0] /= norm[0]

    op = (model/sigma.ravel()[mask.ravel()]).T

    fit, chi = np.linalg.lstsq(op, simage)[:2]
    if (fit<0).any():
        sol = fit
        sol[sol<0] = 1e-11
        bounds = [(1e-11,1e11)]
        result = fmin_slsqp(objf, sol, bounds=bounds, full_output=1, fprime=objdf, acc=1e-19, iter=2000, args=(op.copy(), simage.copy()), iprint=0)
        fit, chi = result[:2]
        fit = np.asarray(fit)
        if (fit<1e-11).any():
            fit[fit<1e-11] = 1e-11

    logp = -0.5*chi - np.log(sigma.ravel()[mask.ravel()]).sum()

    source.amp = fit[0]/norm[0]
    mag = source.Mag(zp)

    if returnImg:
        scaledimg = source.pixeval(xl, yl)
        scaledimg = convolve.convolve(scaledimg, source.convolve, False)[0]

        return logp, mag, scaledimg

    else:
        return logp, mag


def getModel(lens, light, source, scalepar, image, sigma, mask, X, Y, zp=30., returnImg=False):

    source.setPars()
    source.amp = 1.
    lens.setPars()

    simage = ((image/sigma).ravel())[mask.ravel()]

    xl, yl = getDeflections([lens], [X, Y])
    simg = source.pixeval(xl, yl)
    simg = convolve.convolve(simg, source.convolve, False)[0]

    light.setPars()
    light.amp = 1.
    limg = light.pixeval(X, Y)
    limg = convolve.convolve(limg, light.convolve, False)[0]

    if np.isnan(simg).any():
        rimg = 0.*image
        rimg[simg!=simg] = 1.
        return 0., 0., rimg

    model = np.zeros((1, mask.sum()))

    smodel = simg[mask].ravel()
    lmodel = limg[mask].ravel()

    lmag = light.Mag(zp)
    smag = source.Mag(zp)

    rescale = 10.**(-0.4*(lmag + scalepar - smag))

    smodel *= rescale
    model[0] = lmodel + smodel

    norm = model[0].max()

    model[0] /= norm


    op = (model/sigma.ravel()[mask.ravel()]).T

    fit, chi = np.linalg.lstsq(op, simage)[:2]
    if (fit<0).any():
        sol = fit
        sol[sol<0] = 1e-11
        bounds = [(1e-11,1e11)]
        result = fmin_slsqp(objf, sol, bounds=bounds, full_output=1, fprime=objdf, acc=1e-19, iter=2000, args=(op.copy(), simage.copy()), iprint=0)
        fit, chi = result[:2]
        fit = np.asarray(fit)
        if (fit<1e-11).any():
            fit[fit<1e-11] = 1e-11

    logp = -0.5*chi - np.log(sigma.ravel()[mask.ravel()]).sum()

    source.amp = fit[0]*rescale/norm
    light.amp = fit[0]/norm
    mags = (light.Mag(zp), source.Mag(zp))

    if returnImg:
        scaledimg = source.pixeval(xl, yl)
        scaledimg = convolve.convolve(scaledimg, source.convolve, False)[0]
        lsimg = light.pixeval(X, Y)
        lsimg = convolve.convolve(lsimg, light.convolve, False)[0]
        scaledimg += lsimg

        return logp, mags, scaledimg

    else:
        return logp, mags


def getModel_optimizer(lens, light, source, image, sigma, X, Y, zp=30., returnImg=False):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = image.flatten()
    S = sigma.flatten()
# We need to update the parameters of the objects to the values proposed by the optimizer
    light.setPars()
    source.setPars()
    lens.setPars()

    xl,yl = getDeflections(lens,[X,Y])

    model = np.empty((image.size,2))
    model[:,0] = convolve.convolve(light.pixeval(X,Y), light.convolve,False)[0].ravel()/S
    model[:,1] = convolve.convolve(source.pixeval(xl,yl), source.convolve,False)[0].ravel()/S

    amps,chi = optimize.nnls(model,I/S)

    light.amp *= amps[0]
    source.amp *= amps[1]
    if amps[0] <=0.:
        lmag = 99.
    else:
        lmag = light.Mag(zp)
    if amps[1] <= 0.:
        smag = 99.
    else:
        smag = source.Mag(zp)

    logp = -0.5*chi**2

    if returnImg:
        mimage = (amps*(model.T*S).T).sum(axis=1)
        return logp, (lmag, smag), mimage.reshape(image.shape)

    else:
        return logp, amps*(model.T*S).T

