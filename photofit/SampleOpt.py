import numpy
from numpy.random import rand,randn
from math import log,log10

class Sampler:

    def __init__(self,pars,costs,deterministics=[]):
        self.pars = []
        self.deterministics = deterministics
        for par in pars:
            try:
                tmp = par.logp
                self.pars.append(par)
            except:
                self.deterministics.append(par)
        self.costs = costs
        self.nvars = len(self.pars)
        self.cov = None
        self.stretch = None

    def _sample(self,niter):
        nvars = self.nvars

        self.trace = numpy.empty((niter,nvars))
        self.logps = numpy.zeros(niter)
        self.dets = []
        for varIndx in xrange(nvars):
            self.trace[0,varIndx] = self.pars[varIndx].value
            self.logps[0] += self.pars[varIndx].logp
        for cost in self.costs:
            self.logps[0] += cost.logp
        self.dets.append([d.value for d in self.deterministics])

        for i in xrange(1,niter):
            logp = 0.
            proposal = self.trace[i-1]+self.propose()
            bad = False
            for varIndx in xrange(nvars):
                self.pars[varIndx].value = proposal[varIndx]
            for varIndx in xrange(nvars):
                try:
                    logp += self.pars[varIndx].logp
                except:
                    logp += -1e200
                    bad = True
                    break
            if bad==False:
                for cost in self.costs:
                    logp += cost.logp
            self.update(proposal,logp,bad,i)
        for varIndx in xrange(nvars):
            self.pars[varIndx].value = self.trace[-1][varIndx]
    def sample(self,niter):
        self._sample(niter)

    def result(self):
        result = {}
        for i in range(self.nvars):
            result[self.pars[i].__name__] = self.trace[:,i]
        for i in range(len(self.deterministics)):
            d = []
            for item in self.dets:
                d.append(item[i])
            result[self.deterministics[i].__name__] = numpy.array(d)
        return self.logps,self.trace,result

    def setCov(self,cov=None):
        nvars = self.nvars
        if cov is None:
            widths = {'x':0.05,'y':0.05,'reff':0.1,'q':0.03,'pa':5.,'eta':0.03,'nu':0.03,'re':0.1}
            self.cov = numpy.empty(nvars)
            for varIndx in xrange(nvars):
                name = self.pars[varIndx].__name__
                self.cov[varIndx] = widths[name.split('_')[0]]
        else:
            self.cov = numpy.asarray(cov)
        if self.cov.ndim>1:
            self.blank = numpy.zeros(nvars)

    def Z(self):
        if self.stretch is None:
            return 1.
        return 10**(randn(self.nvars)*self.stretch)

    def _propose(self):
        if self.cov is None:
            self.setCov()
        if self.cov.ndim==1:
            return randn(self.nvars)*self.cov*self.Z()
        return numpy.random.multivariate_normal(self.blank,self.cov)*self.Z()
    def propose(self):
        return self._propose()

    def update(self,proposal,logp,bad,i):
        logps = self.logps[i-1]
        if bad==True:
            self.logps[i] = logps
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
            return
        if logp>logps or logp-logps>log(rand()):
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
        else:
            self.logps[i] = logps
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])


class MCMCOpt(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3
        self.setCov(cov)

    def update(self,proposal,logp,bad,i):
        if bad==False and logp>self.logps[i-1]:
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
#            self.cov *= numpy.exp(1./i)
            self.cov /= 1.1
        else:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
#            self.cov /= numpy.exp(1./i)
            self.cov *= 1.1

class AnnealOpt(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3
        self.setCov(cov)
        self.temp = 1.

    def update(self,proposal,logp,bad,i):
        if bad==True:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
            return
        if logp>self.logps[i-1] or logp-self.logps[i-1]>log(rand())*self.temp:
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
            self.temp /= numpy.exp(1./i)
#            self.cov /= numpy.exp(1./i)
        else:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
            self.temp *= numpy.exp(1./i)
#            self.cov *= numpy.exp(1./i)

class Anneal(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3
        self.setCov(cov)
        self.temp = 1.

    def update(self,proposal,logp,bad,i):
        if bad==True:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
            return
        if logp>self.logps[i-1] or logp-self.logps[i-1]>log(rand())*self.temp:
            self.logps[i] = logp
            if logp>=self.logps[:i].max():
                self.trace[i] = proposal
                self.dets.append([d.value for d in self.deterministics])
            else:
                self.trace[i] = self.trace[i-1].copy()
                self.dets.append(self.dets[-1])
        else:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])


class MAOpt(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None,thresh=10):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3
        self.setCov(cov)
        self.nbad = 0
        self.stuck = False
        self.thresh = thresh
        self.temp = 1.

    def update(self,proposal,logp,bad,i):
        logps = self.logps[i-1]
        thresh = self.thresh
        if logp>logps:
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
            self.nbad = 0
            self.stuck = False
            self.temp = 1.
            return
        self.nbad += 1
        if self.nbad>self.thresh and self.stuck==False:
            self.stuck = True
            self.logpTmp = logps
        if self.stuck==True:
            r = log(rand())*self.temp
            print 'stuck',i,logps,self.logpTmp,logp,r,logp-self.logpTmp
            if logp-self.logpTmp>r:
                self.logpTmp = logp
                self.temp /= numpy.exp(1./thresh)
            else:
                self.temp *= numpy.exp(1./thresh)
        self.logps[i] = self.logps[i-1]
        self.trace[i] = self.trace[i-1].copy()
        self.dets.append(self.dets[-1])


class ASA(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3
        self.setCov(cov)
        self.temp = 1.
        self.minprop = 15
        self.naccept = 0
        self.proposals = numpy.zeros((self.minprop,self.nvars))

    def update(self,proposal,logp,bad,i):
        if bad==True:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
            return
        if logp>self.logps[i-1] or logp-self.logps[i-1]>log(rand())*self.temp:
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
            self.temp /= numpy.exp(1./i)
            self.proposals[self.naccept%self.minprop] = proposal-self.trace[i-1]
            self.naccept += 1
            print self.naccept,i
            self.temp /= numpy.exp(1.)
        else:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
            self.temp *= numpy.exp(1./i)

    def propose(self):
        if self.naccept>=self.minprop:
#            indx = numpy.random.randint(0,self.minprop,self.minprop)
            indx = numpy.arange(self.minprop)
            self.cov = numpy.cov(self.proposals[indx].T)
            self.blank = self.proposals[indx].mean(0)
        return self._propose()

class AMCMC(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3
        self.setCov(cov)
        self.minprop = 15
        self.naccept = 0
        self.proposals = numpy.zeros((self.minprop,self.nvars))

    def update(self,proposal,logp,bad,i):
        if bad==False and logp>self.logps[i-1]:
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
#            self.cov *= numpy.exp(1./i)
            self.proposals[self.naccept%self.minprop] = proposal-self.trace[i-1]
            self.naccept += 1
            print self.naccept,i
        else:
            self.logps[i] = self.logps[i-1]
            self.trace[i] = self.trace[i-1].copy()
            self.dets.append(self.dets[-1])
#            self.cov /= numpy.exp(1./i)
#            self.cov *= 1.1

    def propose(self):
        if self.naccept>=self.minprop:
#            indx = numpy.random.randint(0,self.minprop,self.minprop)
            indx = numpy.arange(self.minprop)
            self.cov = numpy.cov(self.proposals[indx].T)
            self.blank = self.proposals[indx].mean(0)
        return self._propose()



class AMAOpt(Sampler):
    def __init__(self,inpars,costs,deterministics,cov=None,thresh=10):
        Sampler.__init__(self,inpars,costs,deterministics)
        self.stretch = 0.3 
        self.setCov(cov)
        self.ocov = self.cov
        self.nbad = 0
        self.stuck = False
        self.thresh = thresh
        self.temp = 1.
        self.minprop = self.nvars*2
        self.naccept = 0
        self.proposals = numpy.zeros((self.minprop,self.nvars))
        self.prop = None
        self.verbose = False

    def set_minprop(self,minprop):
        self.minprop = minprop
        self.proposals = numpy.zeros((self.minprop,self.nvars))

    def update(self,proposal,logp,bad,i):
        logps = self.logps[i-1]
        thresh = self.thresh
        if logp>logps:
            self.logps[i] = logp
            self.trace[i] = proposal
            self.dets.append([d.value for d in self.deterministics])
            self.nbad = 0
            if self.stuck==True:
                self.proposals = self.Oproposals.copy()
                self.naccept = self.Oaccept
            self.proposals[self.naccept%self.minprop] = proposal-self.trace[i-1]
            self.naccept += 1
            if self.naccept>=self.minprop:
                indx = numpy.arange(self.minprop)
                self.cov = numpy.cov(self.proposals[indx].T)
                self.blank = self.proposals[indx].mean(0)

            self.temp = 1.
            self.stuck = False
            if self.verbose:
                print self.naccept,i
            return
        elif self.stuck==False:
            self.prop = None
        self.nbad += 1
        if self.nbad>self.thresh and self.stuck==False:
            self.stuck = True
            self.prop = self.trace[i-1]
            self.logpTmp = logps
            self.Oproposals = self.proposals.copy()
            self.Oaccept = self.naccept
            self.cov = self.ocov
            self.ccount = 0
        if self.stuck==True:
            r = log(rand())*self.temp
            if logp-self.logpTmp>r:
                self.logpTmp = logp
                self.temp /= numpy.exp(1.)
                self.proposals[self.ccount%self.minprop] = proposal-self.prop
                self.prop = proposal
                self.ccount += 1
                if self.ccount>=self.minprop:
                    indx = numpy.arange(self.minprop)
                    indx = numpy.random.randint(0,self.minprop,self.minprop)
                    self.cov = numpy.cov(self.proposals[indx].T)
                    self.blank = self.proposals[indx].mean(0)
                if self.verbose:
                    print self.naccept,i,'stuck',self.logps[i-1],self.logpTmp
            else:
                self.temp *= numpy.exp(1./self.nbad)
        self.logps[i] = self.logps[i-1]
        self.trace[i] = self.trace[i-1].copy()
        self.dets.append(self.dets[-1])

    def propose(self):
#        if self.naccept>=self.minprop:
#            indx = numpy.random.randint(0,self.minprop,self.minprop)
#            indx = numpy.arange(self.minprop)
#            self.cov = numpy.cov(self.proposals[indx].T)
#            self.blank = self.proposals[indx].mean(0)
        return self._propose()


class Opt:
    def __init__(self,pars,costs,deterministics):
        self.pars = []
        for par in pars:
            try:
                tmp = par.logp
                self.pars.append(par)
            except:
                pass
        self.costs = costs
        self.deterministics = deterministics
        self.nvars = len(self.pars)
        self.cov = None
        self.stretch = None

    def opt(self,steps,vals=None,niter=5):
        logp = 0.
        if vals is not None:
            for i in xrange(self.nvars):
                self.pars[i].value = vals[i]
        for par in self.pars:
            logp += par.logp
        for cost in self.costs:
            logp += cost.logp

        tlogp = 0
        for i in xrange(niter):
            if tlogp==logp:
                break
            tlogp = logp
            for j in xrange(self.nvars):
                plogp = 0.
                for par in self.pars:
                    plogp += par.logp
                cplogp = self.pars[j].logp
                plogp -= cplogp
                # Test to step positive or negative
                step = steps[j]
                self.pars[j].value = self.pars[j].value+step
                try:
                    nlogp = plogp+self.pars[j].logp
                except:
                    nlogp = -1e300
                for cost in self.costs:
                    nlogp += cost.logp
                if nlogp<logp:
                    self.pars[j].value = self.pars[j].value-step
                    step *= -1.
                else:
                    logp = nlogp
                while 1:
                    self.pars[j].value = self.pars[j].value+step
                    try:
                        nlogp = plogp+self.pars[j].logp
                    except:
                        self.pars[j].value = self.pars[j].value-step
                        break
                    for cost in self.costs:
                        nlogp += cost.logp
                    if nlogp>logp:
                        logp = nlogp
                    else:
                        self.pars[j].value = self.pars[j].value-step
                        break
        self.logp = logp
        return numpy.array([p.value for p in self.pars])


class levMar:
    def __init__(self,pars,resid,default=1e30):
        from scipy import optimize
        self.orig = numpy.array([p.value for p in pars])
        inpars = numpy.ones(self.orig.size)
        indices = []
        for i in range(inpars.size):
            try:
                tmp = pars[i].logp
                indices.append(i)
            except:
                pass
        def opt(p):
            o = p*self.orig
            for i in indices:
                try:
                    pars[i].value = o[i]
                    tmp = pars[i].logp
                except:
                    return default
            return resid(o)

        coeff,ier = optimize.leastsq(opt,inpars,epsfcn=1e-5)
        self.coeff = coeff*self.orig
        for i in indices:
            pars[i].value = self.coeff[i]



