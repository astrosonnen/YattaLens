def lensModel(inpars,image,sig,gals,lenses,sources,xc,yc,OVRS=1,csub=11,psf=None,noResid=False,verbose=False):
    import pylens,numpy
    import indexTricks as iT

    model = xc*0.
    for gal in gals:
        gal.setPars(inpars)
        model += gal.pixeval(xc,yc,1./OVRS,csub=csub)

    for src in sources:
        src.setPars(inpars)

    for lens in lenses:
        lens.setPars(inpars)

    model = model + pylens.lens_images(lenses,sources,[xc,yc],1./OVRS)
    if numpy.isnan(model.sum()):
        if verbose==True:
            print 'nan model'
        return -1e300

    if OVRS>1:
        model = iT.resamp(model,OVRS,True)

    if psf is not None:
        from imageSim import convolve
        global psfFFT
        if psfFFT is None:
            psf /= psf.sum()
            model,psfFFT = convolve.convolve(model,psf)
        else:
            model,psfFFT = convolve.convolve(model,psfFFT,False)

    if noResid is True:
        return model
    resid = ((model-image)/sig).ravel()
    if verbose==True:
        print "%f  %5.2f %d %dx%d"%((resid**2).sum(),(resid**2).sum()/resid.size,resid.size,image.shape[1],image.shape[0])
    return -0.5*(resid**2).sum()


def lensFit(inpars,image,sig,gals,lenses,sources,xc,yc,OVRS=1,csub=5,psf=None,mask=None,noResid=False,verbose=False,getModel=False,showAmps=False,allowNeg=False):
    import pylens,numpy
    import indexTricks as iT
    from imageSim import convolve
    from scipy import optimize


    if noResid==True or getModel==True:
        mask = None
    if mask is None:
        xin = xc.copy()
        yin = yc.copy()
        imin = image.flatten()
        sigin = sig.flatten()
    else:
        xin = xc[mask].copy()
        yin = yc[mask].copy()
        imin = image[mask].flatten()
        sigin = sig[mask].flatten()

    n = 0
    model = numpy.empty((len(gals)+len(sources),imin.size))
    for gal in gals:
        gal.setPars()
        gal.amp = 1
        if mask is None:
            tmp = gal.pixeval(xin,yin,1./OVRS,csub=csub)
        else:
            tmp = xc*0.
            tmp[mask] = gal.pixeval(xin,yin,1./OVRS,csub=csub)
        if numpy.isnan(tmp).any():
            if verbose==True:
                print 'nan model'
            return -1e300

#        if psf is not None:
#            tmp = convolve.convolve(tmp,psfFFT,False)[0]
        if OVRS>1:
            tmp = iT.resamp(tmp,OVRS,True)
        if psf is not None and gal.convolve is not None:
            tmp = convolve.convolve(tmp,psf,False)[0]
        if mask is None:
            model[n] = tmp.ravel()
        else:
            model[n] = tmp[mask].ravel()
        n += 1

    for lens in lenses:
        lens.setPars()

    x0,y0 = pylens.lens_images(lenses,sources,[xin,yin],1./OVRS,getPix=True)
    for src in sources:
        src.setPars()
        src.amp = 1
        if mask is None:
            tmp = src.pixeval(x0,y0,1./OVRS,csub=csub)
        else:
            tmp = xc*0.
            tmp[mask] = src.pixeval(x0,y0,1./OVRS,csub=csub)
        if numpy.isnan(tmp).any():
            if verbose==True:
                print 'nan model'
            return -1e300

#        if psf is not None:
#            tmp = convolve.convolve(tmp,psfFFT,False)[0]
        if OVRS>1:
            tmp = iT.resamp(tmp,OVRS,True)
        if psf is not None:
            tmp = convolve.convolve(tmp,psf,False)[0]
        if mask is None:
            model[n] = tmp.ravel()
        else:
            model[n] = tmp[mask].ravel()
        n += 1

    rhs = (imin/sigin)
    op = (model/sigin).T

    fit,chi = optimize.nnls(op,rhs)

    if getModel is True:
        j = 0
        for m in gals+sources:
            m.amp = fit[j]
            j += 1
        return (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    elif noResid is True:
        model = (model.T*fit).sum(1).reshape(image.shape)
        j = 0
        for m in gals+sources:
            m.amp = fit[j]
            j += 1
        return model
    model = (model.T*fit).sum(1)

    if mask is None:
        model = model.reshape(image.shape)
        resid = ((model-image)/sig).ravel()
    else:
        resid = (model-imin)/sigin

    if verbose==True:
        print "%f  %5.2f %d %dx%d"%((resid**2).sum(),(resid**2).sum()/resid.size,resid.size,image.shape[1],image.shape[0])
    return -0.5*(resid**2).sum()

