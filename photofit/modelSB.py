from spasmoid.slsqp import fmin_slsqp
#from scipy.optimize import fmin_slsqp
from spasmoid import convolve
import numpy
import indexTricks as iT

def modelSB(inpars,image,sigma,mask,models,xc,yc,OVRS=1,csub=11,noResid=False):
    output = image*0
    for model in models:
        model.setPars(inpars)
        img = model.pixeval(xc,yc)
        if numpy.isnan(img).any() and noResid==False:
            return -1e300
        if model.convolve is not None:
            img = convolve.convolve(img,model.convolve,False)[0]
        output += img

    if noResid==True:
        return output

    logp = (-0.5*((output-image)/sigma)[mask]**2).sum()
    logp -= numpy.log(sigma[mask]).sum()

    return logp


def linearmodelSB(inpars,simage,sigma,mask,models,xc,yc,OVRS=1,csub=11,noResid=False,levMar=False):
    def objf(x,lhs,rhs):
        return ((numpy.dot(lhs,x)-rhs)**2).sum()
    def objdf(x,lhs,rhs):
        return numpy.dot(lhs.T,numpy.dot(lhs,x)-rhs)

    nmod = len(models)
    model = numpy.zeros((nmod,mask.sum()))
    norm = numpy.zeros(nmod)
    for n in range(nmod):
        M = models[n]
        M.setPars(inpars)
        if M.convolve is not None:
            img = M.pixeval(xc,yc,scale=1./abs(OVRS))
            img = convolve.convolve(img,M.convolve,False)[0]
            if OVRS>1:
                img = iT.resamp(img,OVRS,True)
        else:
            img = M.pixeval(xc,yc)
            if OVRS>1:
                img = iT.resamp(img,OVRS,True)
        if numpy.isnan(img).any() and noResid==False:
            return -1e300
        model[n] = img[mask].ravel()
        norm[n] = model[n].max()
        model[n] /= norm[n]

    op = (model/sigma).T
    fit,chi = numpy.linalg.lstsq(op,simage)[:2]
    fit = numpy.array(fit)
    if (fit<0).any():
        sol = fit
        sol[sol<0] = 1e-11
        bounds = [(1e-11,1e11)]*nmod
        result = fmin_slsqp(objf,sol,bounds=bounds,full_output=1,fprime=objdf,acc=1e-19,iter=2000,args=[op.copy(),simage.copy()],iprint=0)
        fit,chi = result[:2]
        fit = numpy.asarray(fit)
        if (fit<1e-11).any():
            fit[fit<1e-11] = 1e-11

    for i in range(nmod):
        models[i].amp = fit[i]/norm[i]

    if levMar==True:
        return (op*fit).sum(1)-simage
        lhs =(op*fit)
        print lhs.shape,simage.shape
        return lhs-simage

    if noResid==True:
        output = []
        for M in models:
            if M.convolve is not None:
                img = M.pixeval(xc,yc,scale=1./abs(OVRS))
                img = convolve.convolve(img,M.convolve,False)[0]
                if OVRS>1:
                    img = iT.resamp(img,OVRS,True)
            else:
                img = M.pixeval(xc,yc)
                if OVRS>1:
                    img = iT.resamp(img,OVRS,True)
            output.append(img)
        return output

    logp = -0.5*chi - numpy.log(sigma).sum()
    return logp

