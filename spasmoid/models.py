from profiles import Sersic as SersicP, Gauss as GaussP, tNIS as tNISP, Hernquist as HerP, Jaffe as JaffeP, sky as skyP
from pointSource import PixelizedModel as PM, GaussianModel as GM
from math import pi

def cnts2mag(cnts,zp):
    from math import log10
    return -2.5*log10(cnts) + zp

_SersicPars = [['amp','n','pa','q','re','x','y'],
                ['logamp','n','pa','q','re','x','y'],
                ['amp','n','q','re','theta','x','y'],
                ['logamp','n','q','re','theta','x','y']]

_tNISPars = [['amp','pa','q','rc','rt','x','y'],
                ['logamp','pa','q','rc','rt','x','y'],
                ['amp','q','rc','rt','theta','x','y'],
                ['logamp','q','rc','rt','theta','x','y']]

_HerPars = [['a','amp','pa','q','x','y'],
                ['a','logamp','pa','q','x','y'],
                ['a','amp','q','theta','x','y'],
                ['a','logamp','q','theta','x','y']]

_JaffePars = [['a','amp','pa','q','x','y'],
                ['a','logamp','pa','q','x','y'],
                ['a','amp','q','theta','x','y'],
                ['a','logamp','q','theta','x','y']]


_skyPars = [['amp'],['logamp'],['amp'],['logamp']]

class Sersic(SersicP):

    def __init__(self,name,var=None,const=None,convolve=0):
        if const is None:
            const = {}
        if var is None:
            var = {}
        # Check for all keys to be set
        keys = var.keys()+const.keys()
        keys.sort()
        if keys not in _SersicPars:
            print "Not all parameters defined!"
            df
        self.invar = var
        self.keys = keys
        self.values = {}
        self.vmap = {}
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        SersicP.__init__(self,x=None,y=None,q=None,pa=None,re=None,amp=None,n=None)
        self.setValues()
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
        elif key=='scale':
            self.__dict__['re'] = value
        else:
            self.__dict__[key] = value


    def setValues(self):
        self.x = self.values['x']
        self.y = self.values['y']
        self.q = self.values['q']
        if 'pa' in self.keys:
            self.pa = self.values['pa']
        else:
            self.theta = self.values['theta']
        self.re = self.values['re']
        if 'amp' in self.keys:
            self.amp = self.values['amp']
        elif self.values['logamp'] is not None:
            self.amp = 10**self.values['logamp']
        self.n = self.values['n']


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

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()


class Gauss(GaussP):
    def __init__(self,name,var=None,const=None,convolve=0):
        if const is None:
            const = {}
        if var is None:
            var = {}
        # Check for all keys to be set
        keys = var.keys()+const.keys()
        keys.sort()
        if 'r0' not in keys:
            keys.append('r0')
            keys.sort()
        if keys!=['amp','pa','q','r0','sigma','x','y']:
            print "Not all parameters defined!"
            df
        self.values = {'r0':None}
        self.vmap = {}
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        GaussP.__init__(self,x=None,y=None,q=None,pa=None,sigma=None,amp=None,r0=None)
        self.setValues()
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
        elif key=='scale':
            self.__dict__['sigma'] = value
        else:
            self.__dict__[key] = value


    def setValues(self):
        self.x = self.values['x']
        self.y = self.values['y']
        self.q = self.values['q']
        self.pa = self.values['pa']
        self.sigma = self.values['sigma']
        self.amp = self.values['amp']
        self.r0 = self.values['r0']


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

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()


class PointSource(GM,PM):
    def __init__(self,name,model,var=None,const=None):
        if const is None:
            const = {}
        if var is None:
            var = {}
        keys = var.keys()+const.keys()
        keys.sort()
        if keys!=['amp','x','y']:
            print "Not all parameters defined!",keys
            df
        self.keys = keys
        self.values = {}
        self.vmap = {}
        self.ispix = False
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        if type(model)==type([]):
            GM.__init__(self,model)
        else:
            PM.__init__(self,model)
            self.ispix = True
        self.setValues()
        self.name = name
        self.convolve = None 

    def __setattr__(self,key,value):
        if key=='logamp':
            if value is not None:
                self.__dict__['amp'] = 10**value
        else:
            self.__dict__[key] = value

    def pixeval(self,xc,yc):
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


class tNIS(tNISP):

    def __init__(self,name,var=None,const=None,convolve=0):
        if const is None:
            const = {}
        if var is None:
            var = {}
        # Check for all keys to be set
        keys = var.keys()+const.keys()
        keys.sort()
        if keys not in _tNISPars:
            print "Not all parameters defined!"
            df
        self.invar = var
        self.keys = keys
        self.values = {}
        self.vmap = {}
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        tNISP.__init__(self,x=None,y=None,q=None,pa=None,rc=None,rt=None,amp=None)
        self.setValues()
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
        elif key=='scale':
            self.__dict__['rt'] = value
        else:
            self.__dict__[key] = value


    def setValues(self):
        self.x = self.values['x']
        self.y = self.values['y']
        self.q = self.values['q']
        if 'pa' in self.keys:
            self.pa = self.values['pa']
        else:
            self.theta = self.values['theta']
        self.rc = self.values['rc']
        self.rt = self.values['rt']
        if 'amp' in self.keys:
            self.amp = self.values['amp']
        elif self.values['logamp'] is not None:
            self.amp = 10**self.values['logamp']


    def getMag(self,amp,zp):
        from math import pi
        rc = self.rc
        rt = self.rt
        cnts = amp*2*pi**2*(rt-rc)
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()

class Hernquist(HerP):

    def __init__(self,name,var=None,const=None,convolve=0):
        if const is None:
            const = {}
        if var is None:
            var = {}
        # Check for all keys to be set
        keys = var.keys()+const.keys()
        keys.sort()
        if keys not in _HerPars:
            print keys
            print "Not all parameters defined!"
            df
        self.invar = var
        self.keys = keys
        self.values = {}
        self.vmap = {}
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        HerP.__init__(self,x=None,y=None,q=None,pa=None,a=None,amp=None)
        self.setValues()
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
        elif key=='scale':
            self.__dict__['a'] = value
        else:
            self.__dict__[key] = value


    def setValues(self):
        self.x = self.values['x']
        self.y = self.values['y']
        self.q = self.values['q']
        if 'pa' in self.keys:
            self.pa = self.values['pa']
        else:
            self.theta = self.values['theta']
        self.a = self.values['a']
        if 'amp' in self.keys:
            self.amp = self.values['amp']
        elif self.values['logamp'] is not None:
            self.amp = 10**self.values['logamp']


    def getMag(self,amp,zp):
        from math import pi
        a = self.a
        cnts = amp
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()


class Jaffe(JaffeP):

    def __init__(self,name,var=None,const=None,convolve=0):
        if const is None:
            const = {}
        if var is None:
            var = {}
        # Check for all keys to be set
        keys = var.keys()+const.keys()
        keys.sort()
        if keys not in _JaffePars:
            print keys
            print "Not all parameters defined!"
            df
        self.invar = var
        self.keys = keys
        self.values = {}
        self.vmap = {}
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        JaffeP.__init__(self,x=None,y=None,q=None,pa=None,a=None,amp=None)
        self.setValues()
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
        elif key=='scale':
            self.__dict__['a'] = value
        else:
            self.__dict__[key] = value


    def setValues(self):
        self.x = self.values['x']
        self.y = self.values['y']
        self.q = self.values['q']
        if 'pa' in self.keys:
            self.pa = self.values['pa']
        else:
            self.theta = self.values['theta']
        self.a = self.values['a']
        if 'amp' in self.keys:
            self.amp = self.values['amp']
        elif self.values['logamp'] is not None:
            self.amp = 10**self.values['logamp']


    def getMag(self,amp,zp):
        from math import pi
        a = self.a
        cnts = amp
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()


class sky(skyP):

    def __init__(self,name,var=None,const=None,convolve=0):
        if const is None:
            const = {}
        if var is None:
            var = {}
        # Check for all keys to be set
        keys = var.keys()+const.keys()
        keys.sort()
        if keys not in _skyPars:
            print keys
            print "Not all parameters defined!"
            df
        self.invar = var
        self.keys = keys
        self.values = {}
        self.vmap = {}
        for key in var.keys():
            self.values[key] = None
            self.vmap[var[key]] = key
        for key in const.keys():
            self.values[key] = const[key]
        skyP.__init__(self,amp=None)
        self.setValues()
        self.name = name
        self.convolve = convolve

    def __setattr__(self,key,value):
        if key=='logamp':
            if value is not None:
                self.__dict__['amp'] = 10**value
        elif key=='scale':
            self.__dict__['a'] = value
        else:
            self.__dict__[key] = value


    def setValues(self):
        if 'amp' in self.keys:
            self.amp = self.values['amp']
        elif self.values['logamp'] is not None:
            self.amp = 10**self.values['logamp']


    def getMag(self,amp,zp):
        from math import pi
        cnts = amp
        return cnts2mag(cnts,zp)

    def Mag(self,zp):
        return self.getMag(self.amp,zp)

    def setPars(self,pars):
        for key in self.vmap:
            self.values[self.vmap[key]] = pars[key]
        self.setValues()
