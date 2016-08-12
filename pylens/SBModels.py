import SBProfiles as SBProfiles
from math import pi

def cnts2mag(cnts,zp):
    from math import log10
    return -2.5*log10(cnts) + zp


class SBModel:
    def __init__(self,name,pars,convolve=0):
        if 'amp' not in pars.keys() and 'logamp' not in pars.keys():
            pars['amp'] = 1.
        self.keys = pars.keys()
        self.keys.sort()
        if self.keys not in self._SBkeys:
            import sys
            print 'Not all (or too many) parameters were defined!'
            sys.exit()
        self._baseProfile.__init__(self)
        self.vmap = {}
        self.pars = pars
        for key in self.keys:
            try:
                v = self.pars[key].value
                self.vmap[key] = self.pars[key]
            except:
                self.__setattr__(key,self.pars[key])
        self.setPars()
        self.name = name
        self.convolve = convolve


    def __setattr__(self,key,value):
        if key=='pa':
            self.__dict__['pa'] = value
            if value is not None:
                self.__dict__['theta'] = value*pi/180.
        elif key=='theta':
            if value is not None:
                self.__dict__['pa'] = value*180./pi
            self.__dict__['theta'] = value
        elif key=='logamp':
            if value is not None:
                self.__dict__['amp'] = 10**value
        else:
            self.__dict__[key] = value


    def setPars(self):
        for key in self.vmap:
            self.__setattr__(key,self.vmap[key].value)


class Sersic(SBModel,SBProfiles.Sersic):
    _baseProfile = SBProfiles.Sersic
    _SBkeys = [['amp','n','pa','q','re','x','y'],
                ['logamp','n','pa','q','re','x','y'],
                ['amp','n','q','re','theta','x','y'],
                ['logamp','n','q','re','theta','x','y']]

    def __init__(self,name,pars,convolve=0):
        SBModel.__init__(self,name,pars,convolve)

    def getMag(self,amp,zp):
        from scipy.special import gamma
        from math import exp,pi
        n = self.n
        re = self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        cnts = (re**2)*amp*exp(k)*n*(k**(-2*n))*gamma(2*n)*2*pi
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)


class Sersic_wboxyness(SBModel, SBProfiles.Sersic_wboxyness):
    _baseProfile = SBProfiles.Sersic_wboxyness
    _SBkeys = [['amp','b4', 'n','pa','q','re','x','y'],
                ['b4', 'logamp','n','pa','q','re','x','y'],
                ['amp','b4', 'n','q','re','theta','x','y'],
                ['b4', 'logamp','n','q','re','theta','x','y']]

    def __init__(self,name,pars,convolve=0):
        SBModel.__init__(self,name,pars,convolve)

    def getMag(self,amp,zp):
        from scipy.special import gamma
        from math import exp,pi
        n = self.n
        re = self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        cnts = (re**2)*amp*exp(k)*n*(k**(-2*n))*gamma(2*n)*2*pi
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)


class Spiral(SBModel, SBProfiles.Spiral):
    _baseProfile = SBProfiles.Spiral
    _SBkeys = [['amp', 'bar', 'disk', 'h', 'omega', 'pa', 'q', 'ra', 'x', 'y']]

    def __init__(self, name, pars, convolve=0):
        SBModel.__init__(self, name, pars, convolve)

    def getMag(self, amp, zp):
	cnts = amp
        return cnts2mag(cnts, zp)

    def Mag(self, zp):
        return self.getMag(self.amp, zp)


class Ring(SBModel, SBProfiles.Ring):
    _baseProfile = SBProfiles.Ring
    _SBkeys = [['amp', 'hi', 'ho', 'pa', 'q', 'rr', 'x', 'y']]

    def __init__(self, name, pars, convolve=0):
        SBModel.__init__(self, name, pars, convolve)

    def getMag(self, amp, zp):
	cnts = amp
        return cnts2mag(cnts, zp)

    def Mag(self, zp):
        return self.getMag(self.amp, zp)


class StoneRing(SBModel, SBProfiles.StoneRing):
    _baseProfile = SBProfiles.StoneRing
    _SBkeys = [['amp', 'omega', 'pa', 'q', 'rr', 'smooth', 'spa', 'stone', 'width', 'x', 'y']]

    def __init__(self, name, pars, convolve=0):
        SBModel.__init__(self, name, pars, convolve)

    def getMag(self, amp, zp):
	cnts = amp
        return cnts2mag(cnts, zp)

    def Mag(self, zp):
        return self.getMag(self.amp, zp)


class Arc(SBModel, SBProfiles.Arc):
    _baseProfile = SBProfiles.Arc
    _SBkeys = [['amp', 'hr', 'ht', 'invrc', 'length', 'pa', 'x', 'y']]

    def __init__(self, name, pars, convolve=0):
        SBModel.__init__(self, name, pars, convolve)

    def getMag(self, amp, zp):
        cnts = amp
        return cnts2mag(cnts, zp)

    def Mag(self, zp):
        return self.getMag(self.amp, zp)

class Gauss(SBModel,SBProfiles.Gauss):
    _baseProfile = SBProfiles.Gauss
    _SBkeys = [['amp','pa','q','r0','sigma','x','y']]

    def __init__(self,name,pars,convolve=0):
        if 'r0' not in pars.keys():
            pars['r0'] = None
        SBModel.__init__(self,name,pars,convolve)

    def getMag(self,amp,zp):
        from math import exp,pi
        if self.r0 is None:
            cnts = amp/(2*pi*self.sigma**2)
        else:
            from scipy.special import erf
            r0 = self.r0
            s = self.sigma
            r2pi = (2*pi)**0.5
            cnts = amp*pi*s*(r2pi*r0*(1.+erf(r0/(s*2**0.5)))+2*s*exp(-0.5*r0**2/s**2))
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)



class BAHBA:
    def __setattr__(self,key,value):
        if key=='logamp':
            if value is not None:
                self.__dict__['amp'] = 10**value
        else:
            self.__dict__[key] = value

    def pixeval(self,xc,yc,dummy1=None,dummy2=None,**kwargs):
        if self.ispix==True:
            return PM.pixeval(self,xc,yc)
        else:
            return GM.pixeval(self,xc,yc)

    def setValues(self):
        self.x = self.values['x']
        self.y = self.values['y']
        if 'amp' in self.keys:
            self.amp = self.values['amp']
        elif self.values['logamp'] is not None:
            self.amp = 10**self.values['logamp']

    def getMag(self,amp,zp):
        return cnts2mag(amp,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()

