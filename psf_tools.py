import numpy as np
from scipy.signal import convolve2d
from scipy.optimize import leastsq


def fit_gaussian(psf):

    ny, nx = psf.shape

    x0 = (nx-1.)/2.
    y0 = (ny-1.)/2.

    x = np.arange(nx)
    y = np.arange(ny)

    Y, X = np.meshgrid(y, x)

    def model(p):

        cov = np.array(((p[3], p[4]), (p[4], p[5])))
        invcov = 1./np.linalg.det(cov)*np.array(((cov[1, 1], -cov[0, 1]), (-cov[1, 0], cov[0, 0])))

        prod = ((X-p[1])*invcov[0, 0] + (Y-p[2])*invcov[1, 0])*(X-p[1]) + ((X-p[1])*invcov[0, 1] + (Y-p[2])*invcov[1, 1])*(Y-p[2])

        return p[0]/(2.*np.pi)/np.linalg.det(cov)**0.5*np.exp(-0.5*prod)

    def errfunc(p):

        mod = model(p)

        return (mod - psf).flatten()

    guess = (psf.sum(), x0, y0, 3., 0., 3.)

    p = leastsq(errfunc, guess)[0]

    return p


def get_kernel(psf1, psf2, cut=5):

    ft1 = np.fft.fft2(psf1)
    ft2 = np.fft.fft2(psf2)

    ny, nx = psf1.shape

    Ktilde = ft1/ft2

    kx = np.hstack((np.arange((nx+1)/2), np.arange(nx/2) - nx/2))
    ky = np.hstack((np.arange((ny+1)/2), np.arange(ny/2) - ny/2))

    KY_full, KX_full = np.meshgrid(ky, kx)

    kcut = np.hstack((np.arange(cut+1), np.arange(cut) - cut))
    KYcut, KXcut = np.meshgrid(kcut, kcut)

    Ktilde_tofit = np.vstack((np.hstack((Ktilde[:cut+1,:cut+1], Ktilde[:cut+1, -cut:])), \
                              np.hstack((Ktilde[-cut:, :cut+1], Ktilde[-cut:, -cut:]))))

    def model(KX, KY, p):
        cov = np.array(((p[3], p[4]), (p[4], p[5])))
        prod = (KX*cov[0, 0] + KY*cov[1, 0])*KX + (KX*cov[0, 1] + KY*cov[1, 1])*KY

        return p[0]*np.exp(-2.*np.pi**2*prod/float(nx)**2)*\
               np.exp(-2.*np.pi*1j*KX*(p[1] - nx/2)/float(nx))*\
               np.exp(-2.*np.pi*1j*KY*(p[2] - ny/2)/float(nx))

    def absonly_model(KX, KY, p):
        prod = (KX*p[1] + KY*p[2])*KX + (KX*p[2] + KY*p[3])*KY

        return p[0]*np.exp(-2.*np.pi**2*prod/float(nx)**2)

    def errfunc(p):
        mod = absonly_model(KXcut, KYcut, p)

        #out = np.hstack((np.real(mod) - np.real(Ktilde_tofit), np.imag(mod) - np.imag(Ktilde_tofit))).flatten()
        out = (np.abs(Ktilde_tofit) - mod).flatten()

        return out

    amp_guess = np.abs(Ktilde).sum()/float(nx)/float(ny)*(2.*np.pi)
    #guess = (amp_guess, nx/2, ny/2, 1., 0., 1.)
    guess = (amp_guess, 1., 0., 1.)

    p = leastsq(errfunc, guess)[0]

    cov = np.array(((p[1], p[2]), (p[2], p[3])))
    invcov = 1./np.linalg.det(cov)*np.array(((cov[1, 1], -cov[0, 1]), (-cov[1, 0], cov[0, 0])))

    x = np.arange(nx)
    y = np.arange(ny)
    x0 = float(nx/2)
    y0 = float(ny/2)

    Y, X = np.meshgrid(y, x)

    prod = ((X - x0)*invcov[0, 0] + (Y - y0)*invcov[1, 0])*(X - x0) + \
           ((X - x0)*invcov[0, 1] + (Y - y0)*invcov[1, 1])*(Y - y0)

    gauss = p[0]/(2.*np.pi)/np.linalg.det(cov)**0.5*np.exp(-0.5*prod)

    """
    full_model = model(KX_full, KY_full, p)
    full_model = absonly_model(KX_full, KY_full, (p[0], p[3], p[4], p[5]))
    Ktilde[cut+1:-cut] = full_model[cut+1:-cut]
    Ktilde[:, cut+1:-cut] = full_model[:, cut+1:-cut]
    Ktilde = full_model

    K = np.fft.ifft2(Ktilde*np.exp(-np.pi*1j*(KY_full + KX_full)*(nx-1)/float(nx)))

    return K
    """

    return gauss


def match_psf(psf1, psf2):
    # matches psf1 (narrower) to psf2 (broader)

    K = get_kernel(psf2, psf1)

    matched_psf1 = convolve2d(psf1, np.abs(K))

    ny, nx = psf1.shape

    matched_psf1 = matched_psf1[ny/2:-ny/2+1, nx/2:-nx/2+1]

    return matched_psf1


