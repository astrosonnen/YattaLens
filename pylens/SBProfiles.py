import numpy as np
from scipy import interpolate
"""
MINIMAL ERROR CHECKING!
"""
def cnts2mag(cnts,zp):
    from math import log10
    return -2.5*log10(cnts) + zp


class Sersic:
    def __init__(self,x=None,y=None,q=None,pa=None,re=None,amp=None,n=None):
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.re = re
        self.amp = amp
        self.n = n
        self.convolve = True

    def setAmpFromMag(self,mag,zp):
        from math import exp,log10,pi
        from scipy.special import gamma
        cnts = 10**(-0.4*(mag-zp))
        n = self.n
        re = self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        self.amp = cnts/((re**2)*exp(k)*n*(k**(-2*n))*gamma(2*n)*2*pi)

    def eval(self,r):
        k = 2.*self.n-1./3+4./(405.*self.n)+46/(25515.*self.n**2)
        R = r/self.re
        return self.amp*np.exp(-k*(R**(1./self.n) - 1.))

    def pixeval(self,x,y,scale=1,csub=23):
        from scipy import interpolate
        from math import pi,cos as COS,sin as SIN
        shape = x.shape
        x = x.ravel()
        y = y.ravel()

        cos = COS(self.pa*pi/180.)
        sin = SIN(self.pa*pi/180.)
        xp = (x-self.x)*cos+(y-self.y)*sin
        yp = (y-self.y)*cos-(x-self.x)*sin
        r = (self.q*xp**2+yp**2/self.q)**0.5

        k = 2.*self.n-1./3+4./(405.*self.n)+46/(25515.*self.n**2)
        R = np.logspace(-5.,4.,451) # 50 pnts / decade
        s0 = np.exp(-k*(R**(1./self.n) - 1.))

        # Determine corrections for curvature
        rpow = R**(1./self.n - 1.)
        term1 = (k*rpow/self.n)**2
        term2 = k*(self.n-1.)*rpow/(R*self.n**2)
        wid = scale/self.re
        corr = (term1+term2)*wid**3/6.
        try:
            minR = R[abs(corr)<0.005].min()
        except:
            minR = 0

        # Evaluate model!
        model = interpolate.splrep(R,s0,k=3,s=0)
        R0 = r/self.re
        s = interpolate.splev(R0,model)*scale**2
        if self.n<=1. or minR==0:
            return self.amp*s.reshape(shape)
        model2 = interpolate.splrep(R,s0*R*self.re**2,k=3,s=0)
        coords = np.where(R0<minR)[0]
        c = (np.indices((csub,csub)).astype(np.float32)-csub/2)*scale/csub
        for i in coords:
            # The central pixels are tricky because we can't assume that we
            #   are integrating in delta-theta segments of an annulus; these
            #   pixels are treated separately by sub-sampling with ~500 pixels
            if R0[i]<3*scale/self.re:
                s[i] = 0.
                y0 = c[1]+y[i]
                x0 = c[0]+x[i]
                xp = (x0-self.x)*cos+(y0-self.y)*sin
                yp = (y0-self.y)*cos-(x0-self.x)*sin
                r0 = (self.q*xp**2+yp**2/self.q)**0.5/self.re
                s[i] = interpolate.splev(r0.ravel(),model).mean()*scale**2
                continue
            lo = R0[i]-0.5*scale/self.re
            hi = R0[i]+0.5*scale/self.re
            angle = (scale/self.re)/R0[i]
            s[i] = angle*interpolate.splint(lo,hi,model2)
            # The following code should no longer be needed
            """
            if lo<0:
                s[i] = ((interpolate.splint(0,abs(lo),model2)+interpolate.splint(0,hi,model2)))*pi*2
            else:
                s[i] = angle*interpolate.splint(lo,hi,model2)
            """
        return self.amp*s.reshape(shape)


class Sersic_wboxyness:
    def __init__(self, x=None, y=None, q=None, pa=None, re=None, amp=None, n=None, b4=None):
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.re = re
        self.amp = amp
        self.n = n
        self.b4 = b4
        self.convolve = True

    def setAmpFromMag(self,mag,zp):
        from math import exp,log10,pi
        from scipy.special import gamma
        cnts = 10**(-0.4*(mag-zp))
        n = self.n
        re = self.re
        k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
        self.amp = cnts/((re**2)*exp(k)*n*(k**(-2*n))*gamma(2*n)*2*pi)

    def eval(self,r):
        k = 2.*self.n-1./3+4./(405.*self.n)+46/(25515.*self.n**2)
        R = r/self.re
        return self.amp*np.exp(-k*(R**(1./self.n) - 1.))

    def pixeval(self,x,y,scale=1,csub=23):
        from scipy import interpolate
        from math import pi,cos as COS,sin as SIN
        shape = x.shape
        x = x.ravel()
        y = y.ravel()

        cos = COS(self.pa*pi/180.)
        sin = SIN(self.pa*pi/180.)

        xp = (x-self.x)*cos+(y-self.y)*sin
        yp = (y-self.y)*cos-(x-self.x)*sin
        r = (self.q*xp**2+yp**2/self.q)**0.5

        cosp = xp*self.q**0.5/r

        cos4p = 2.*(2.*cosp**2 - 1.)**2 - 1.

        r *= (1. + self.b4*cos4p)

        k = 2.*self.n-1./3+4./(405.*self.n)+46/(25515.*self.n**2)
        R = np.logspace(-5.,4.,451) # 50 pnts / decade
        s0 = np.exp(-k*(R**(1./self.n) - 1.))

        # Determine corrections for curvature
        rpow = R**(1./self.n - 1.)
        term1 = (k*rpow/self.n)**2
        term2 = k*(self.n-1.)*rpow/(R*self.n**2)
        wid = scale/self.re
        corr = (term1+term2)*wid**3/6.
        try:
            minR = R[abs(corr)<0.005].min()
        except:
            minR = 0

        # Evaluate model!
        model = interpolate.splrep(R,s0,k=3,s=0)
        R0 = r/self.re
        s = interpolate.splev(R0,model)*scale**2
        if self.n<=1. or minR==0:
            return self.amp*s.reshape(shape)
        model2 = interpolate.splrep(R,s0*R*self.re**2,k=3,s=0)
        coords = np.where(R0<minR)[0]
        c = (np.indices((csub,csub)).astype(np.float32)-csub/2)*scale/csub
        for i in coords:
            # The central pixels are tricky because we can't assume that we
            #   are integrating in delta-theta segments of an annulus; these
            #   pixels are treated separately by sub-sampling with ~500 pixels
            if R0[i]<3*scale/self.re:
                s[i] = 0.
                y0 = c[1]+y[i]
                x0 = c[0]+x[i]
                xp = (x0-self.x)*cos+(y0-self.y)*sin
                yp = (y0-self.y)*cos-(x0-self.x)*sin
                r0 = (self.q*xp**2+yp**2/self.q)**0.5/self.re
                s[i] = interpolate.splev(r0.ravel(),model).mean()*scale**2
                continue
            lo = R0[i]-0.5*scale/self.re
            hi = R0[i]+0.5*scale/self.re
            angle = (scale/self.re)/R0[i]
            s[i] = angle*interpolate.splint(lo,hi,model2)
            # The following code should no longer be needed
            """
            if lo<0:
                s[i] = ((interpolate.splint(0,abs(lo),model2)+interpolate.splint(0,hi,model2)))*pi*2
            else:
                s[i] = angle*interpolate.splint(lo,hi,model2)
            """
        return self.amp*s.reshape(shape)


class Ring:
    def __init__(self, x=None, y=None, q=None, pa=None, rr=None, amp=None, hi=None, ho=None):
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.rr = rr
        self.amp = amp
        self.hi = hi
        self.ho = ho
        self.convolve = True

    def pixeval(self, x, y):
        from math import pi,cos as COS,sin as SIN
        shape = x.shape
        x = x.ravel()
        y = y.ravel()

        cos = COS(self.pa*pi/180.)
        sin = SIN(self.pa*pi/180.)
        xp = (x-self.x)*cos+(y-self.y)*sin
        yp = (y-self.y)*cos-(x-self.x)*sin
        r = (self.q*xp**2+yp**2/self.q)**0.5

        inner = 0.5*(np.sign(self.rr - r) + 1.)*np.exp(-(self.rr - r)/self.hi)
        outer = 0.5*(np.sign(r - self.rr) + 1.)*np.exp(-(r - self.rr)/self.ho)

        return self.amp*(inner + outer).reshape(shape)


class Spiral:
    def __init__(self, x=None, y=None, q=None, pa=None, ra=None, omega=None, amp=None, h=None, bar=None, disk=None):

        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.ra = ra
        self.amp = amp
        self.omega = omega
        self.h = h
        self.bar = bar
        self.disk = disk
        self.convolve = True

    def pixeval(self, x, y):
        from math import pi,cos as COS,sin as SIN
        shape = x.shape
        x = x.ravel()
        y = y.ravel()

        cos = COS(self.pa*pi/180.)
        sin = SIN(self.pa*pi/180.)
        tan = np.tan(self.pa*pi/180.)

        xp = (x-self.x)*cos+(y-self.y)*sin
        yp = (y-self.y)*cos-(x-self.x)*sin
        r = (self.q*xp**2+yp**2/self.q)**0.5

        cospix = xp/r*self.q**0.5
        sinpix = yp/r/self.q**0.5
        tanpix = yp/xp/self.q

        cosdiff = cos*cospix + sin*sinpix
        sindiff = sin*cospix - cos*sinpix
        tandiff = (tan - tanpix)/(1. + tan*tanpix)

        angdiffa = np.arccos(cospix) * np.sign(yp) * 180./pi
        angdiffb = -np.arccos(-cospix) * np.sign(yp) * 180./pi
        #angdiff = np.arcsin(sindiff)
        #angdiff = np.arctan(tandiff)

        arma = (angdiffa - self.omega)*angdiffa < 0.
        edgebara = angdiffa*self.omega < 0.
        edgearma = (angdiffa - self.omega)*self.omega > 0.

        armb = (angdiffb - self.omega)*angdiffb < 0.
        edgebarb = angdiffb*self.omega < 0.
        edgearmb = (angdiffb - self.omega)*self.omega > 0.

        dr_a = np.inf + 0.*x
        dr_b = np.inf + 0.*x

        dr_a[arma] = np.abs(r[arma] - self.ra)
        dr_a[edgebara] = ((r[edgebara] - self.ra)**2 + self.ra**2*(angdiffa[edgebara]/180.*pi)**2)**0.5
        dr_a[edgearma] = ((r[edgearma] - self.ra)**2 + self.ra**2*((angdiffa[edgearma] - self.omega)/180.*pi)**2)**0.5

        dr_b[armb] = np.abs(r[armb] - self.ra)
        dr_b[edgebarb] = ((r[edgebarb] - self.ra)**2 + self.ra**2*(angdiffb[edgebarb]/180.*pi)**2)**0.5
        dr_b[edgearmb] = ((r[edgearmb] - self.ra)**2 + self.ra**2*((angdiffb[edgearmb] - self.omega)/180.*pi)**2)**0.5

        sb = self.amp*(np.exp(-dr_a/self.h) + np.exp(-dr_b/self.h) + self.disk*np.exp(-r/self.ra))

        return sb.reshape(shape)


class StoneRing:
    def __init__(self, x=None, y=None, amp=None, rr=None, pa=None, q=None, width=None, smooth=1., spa=None, \
                 omega=None, stone=None):

        self.x = x
        self.y = y
        self.rr = rr
        self.width = width
        self.pa = pa
        self.q = q
        self.smooth = smooth
        self.spa = spa
        self.omega = omega
        self.stone = stone
        self.amp = amp
        self.convolve = True

    def pixeval(self, x, y):
        from math import pi,cos as COS,sin as SIN
        shape = x.shape
        x = x.ravel()
        y = y.ravel()

        cos = COS(self.pa*pi/180.)
        sin = SIN(self.pa*pi/180.)
        xp = (x-self.x)*cos+(y-self.y)*sin
        yp = (y-self.y)*cos-(x-self.x)*sin
        r = (self.q*xp**2+yp**2/self.q)**0.5

        inner = 0.5*(np.sign(self.rr - 0.5*self.width - r) + 1.)*np.exp(-(self.rr - 0.5*self.width - r)/self.smooth)
        outer = 0.5*(np.sign(r - self.rr - 0.5*self.width) + 1.)*np.exp(-(r - self.rr - 0.5*self.width)/self.smooth)
        body = 0.5*(np.sign(r - self.rr + 0.5*self.width) + 1.)*0.5*(np.sign(self.rr + 0.5*self.width - r) + 1.)

        ring = self.amp*(inner + outer + body)

        """
        stonecos = COS(self.spa*pi/180.)
        stonesin = SIN(self.spa*pi/180.)

        scosp = stonecos*cos + stonesin*sin
        ssinp = stonesin*cos - stonecos*sin

        cospix = xp/r*self.q**0.5
        sinpix = yp/r/self.q**0.5

        cosdiff = scosp*cospix + ssinp*sinpix

        angdiff = np.arccos(cosdiff)*180./pi

        stonebody = 0.5*(np.sign(0.5*self.omega - angdiff) + 1.)
        stoneedge = 0.5*(np.sign(angdiff - 0.5*self.omega) + 1.) * np.exp(-r*abs(angdiff - 0.5*self.omega)*pi/180./self.smooth)
        stonesmooth = np.exp(-abs(angdiff - 0.5*self.omega)/90.)

        #ring *= 1. + self.stone*(stonebody + stoneedge)
        ring *= 1. + self.stone*stonesmooth
        """

        return ring.reshape(shape)


class Arc:
    def __init__(self, x=None, y=None, length=None, pa=None, hr=None, ht=None, amp=None, invrc=None):
        self.x = x
        self.y = y
        self.length = length
        self.pa = pa
        self.hr = hr
        self.ht = ht
        self.invrc = invrc
        self.amp = amp
        self.convolve = True

    def rc(self):
        return 1./self.invrc

    def get_center(self):
        from math import pi
        cos = np.cos(0.5*pi*(np.sign(-self.rc()) + 1.) + self.pa*pi/180.)
        sin = np.sin(0.5*pi*(np.sign(-self.rc()) + 1.) + self.pa*pi/180.)

        xc = self.x - np.abs(self.rc())*cos
        yc = self.y - np.abs(self.rc())*sin

        return (xc, yc)

    def pixeval(self, x, y, scale=1, csub=23):
        from scipy import interpolate
        from math import pi,cos as COS,sin as SIN
        shape = x.shape
        x = x.ravel()
        y = y.ravel()

        cos = COS(0.5*pi*(np.sign(-self.rc()) + 1.) + self.pa*pi/180.)
        sin = SIN(0.5*pi*(np.sign(-self.rc()) + 1.) + self.pa*pi/180.)
        tan = np.tan(0.5*pi*(np.sign(-self.rc()) + 1.) + self.pa*pi/180.)

        xc = self.x - np.abs(self.rc())*cos
        yc = self.y - np.abs(self.rc())*sin

        r = ((x - xc)**2 + (y - yc)**2)**0.5

        cospix = (x - xc)/r
        sinpix = (y - yc)/r
        tanpix = (y - yc)/(x - xc)

        cosdiff = cos*cospix + sin*sinpix
        sindiff = sin*cospix - cos*sinpix
        tandiff = (tan - tanpix)/(1. + tan*tanpix)

        angdiff = np.arccos(cosdiff)

        cut = 0.5*self.length*np.abs(self.invrc)
        body = angdiff < cut
        edge = np.logical_not(body)

        dr = 0.*x

        q = self.hr/self.ht

        dr[body] = np.abs(r[body] - np.abs(self.rc()))/self.hr

        dr[edge] = ((self.rc()*(angdiff[edge] - cut))**2/self.ht**2 + np.abs(r[edge] - np.abs(self.rc()))**2/self.hr**2)**0.5

        #dr = np.abs(r - self.rc)
        #dt = np.abs(angdiff - cut)*r
        #rad = self.amp*np.exp(-dr/self.h)
        #core = self.amp*0.5*(np.sign(angdiff + cut) + 1.)*0.5*(np.sign(cut - angdiff) + 1.)
        #edge = 0.5*self.amp*(np.sign(angdiff - cut) + 2. + np.sign(-cut - angdiff)) * np.exp(-dt/self.ht)
        #return (rad*(core + edge)).reshape(shape)

        return (self.amp*np.exp(-dr)).reshape(shape)

    def get_edges(self):
        from math import pi
        theta = 0.5*pi*(np.sign(-self.rc()) + 1.) + self.pa*pi/180.
        theta_a = theta + 0.5*self.length*np.abs(self.invrc)
        theta_b = theta - 0.5*self.length*np.abs(self.invrc)

        cos = np.cos(theta)
        sin = np.sin(theta)

        xc = self.x - np.abs(self.rc())*cos
        yc = self.y - np.abs(self.rc())*sin

        xa = xc + np.abs(self.rc())*np.cos(theta_a)
        ya = yc + np.abs(self.rc())*np.sin(theta_a)

        xb = xc + np.abs(self.rc())*np.cos(theta_b)
        yb = yc + np.abs(self.rc())*np.sin(theta_b)

        return ((xa, ya), (xb, yb))


class deV(Sersic):

    def __init__(self,x=None,y=None,q=None,pa=None,re=None,amp=None):
        Sersic.__init__(self,x,y,q,pa,re,amp,4.)


class exp(Sersic):

    def __init__(self,x=None,y=None,q=None,pa=None,re=None,amp=None):
        Sersic.__init__(self,x,y,q,pa,re,amp,1.)


class Gauss:

    def __init__(self,x=None,y=None,q=None,pa=None,sigma=None,amp=None,r0=None):
        self.x = x
        self.y = y
        self.q = q
        self.pa = pa
        self.sigma = sigma
        self.amp = amp
        self.r0 = r0
        self.convolve = True

    def pixeval(self,x,y,factor=None,csub=None):
        from math import pi

        cos = np.cos(self.pa*pi/180.)
        sin = np.sin(self.pa*pi/180.)
        xp = (x-self.x)*cos+(y-self.y)*sin
        yp = (y-self.y)*cos-(x-self.x)*sin
        r2 = (self.q*xp**2+yp**2/self.q)
        if self.r0 is None:
            return self.amp*np.exp(-0.5*r2/self.sigma**2)
        return self.amp*np.exp(-0.5*(r2**0.5-self.r0)**2/self.sigma**2)


    def getMag(self,zp):
        from math import exp,pi
        if self.r0 is None:
            cnts = self.amp*(2*pi*self.sigma**2)
        else:
            from scipy.special import erf
            r0 = self.r0
            s = self.sigma
            r2pi = (2*pi)**0.5
            cnts = self.amp*pi*s*(r2pi*r0*(1.+erf(r0/(s*2**0.5)))+2*s*exp(-0.5*r0**2/s**2))
        return cnts2mag(cnts,zp)


    def eval(self,x,y):
        from math import pi
        try:
            cos = np.cos(self.theta)
            sin = np.sin(self.theta)
            xp = (x-self.x)*cos+(y-self.y)*sin
            yp = (y-self.y)*cos-(x-self.x)*sin
            r = (self.q*xp**2+yp**2/self.q)**0.5/self.sigma
            s = self.amp*np.exp(-0.5*r**self.n)/(2.*pi*self.sigma**2)**1.0
            return s
        except:
            return x*0.

