from pylens import MassModels
import numpy,pylab
import indexTricks as iT
from scipy import interpolate
from scipy.special import gamma,gammainc
import ndinterp,time
import sersic
import cPickle

x0 = numpy.logspace(-4.,1.,51)
n0 = numpy.linspace(0.5,6.,12)
q0 = numpy.linspace(0.1,1.,19)

x,y,n,q = ndinterp.create_axes_array([x0,x0,n0,q0])
yout = x*0.
xout = y*0.
for i in range(x.shape[2]):
    for j in range(x.shape[3]):
        X = x[:,:,i,j]
        Y = y[:,:,i,j]
        N = n[0,0,i,j]
        Q = q[0,0,i,j]
        k = 2.*N-1./3+4./(405.*N)+46/(25515.*N**2)
        amp = k**(2*N)/(2*N*gamma(2*N))
        yi,xi = sersic.sersicdeflections(-Y.ravel(),X.ravel(),amp,1.,N,Q)
        yout[:,:,i,j] = -1*yi.reshape(Y.shape)
        xout[:,:,i,j] = xi.reshape(X.shape)

axes = {}
axes[0] = interpolate.splrep(x0,numpy.arange(x0.size))
axes[1] = interpolate.splrep(x0,numpy.arange(x0.size))
axes[2] = interpolate.splrep(n0,numpy.arange(n0.size))
axes[3] = interpolate.splrep(q0,numpy.arange(q0.size))

xmodel = ndinterp.ndInterp(axes,xout)
ymodel = ndinterp.ndInterp(axes,yout)

f = open('serModelsHDR.dat','wb')
cPickle.dump([xmodel,ymodel],f,2)
f.close()
