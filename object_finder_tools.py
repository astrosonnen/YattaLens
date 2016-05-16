import pyfits
import numpy as np
from scipy.optimize import minimize

def compare_colors(f1a, f1b, f2a, f2b, e1a, e1b, e2a, e2b, nsigma=4.):
    """
    Compares fluxes of two objects, 1 and 2, in two bands, a and b, and determines whether they have the same color
    :param f1a: band a flux of object 1
    :param f1b: band b flux of object 1
    :param f2a: band a flux of object 2
    :param f2b: band b flux of object 2
    :param e1a: measurement error on f1a
    :param e1b: measurement error on f1b
    :param e2a: measurement error on f2a
    :param e2b: measurement error on f2b
    :param nsigma: number of sigmas required to reject the null hypothesis of identical color
    :return: consistent: bool-type
    """

    #if f1a > nsigma*e1a and f1b > nsigma*e1b and f2a > nsigma*e2a and f2b > nsigma*e2b:
    def nloglike(p):
        f1, f2, c = p
        #t1a = -0.5*np.log(2.*np.pi) - np.log(e1a) - 0.5*(f1a - f1)**2/e1a**2
        #t1b = -0.5*np.log(2.*np.pi) - np.log(e1b) - 0.5*(f1a - f1*c)**2/e1b**2
        #t2a = -0.5*np.log(2.*np.pi) - np.log(e2a) - 0.5*(f2a - f2)**2/e2a**2
        #t2b = -0.5*np.log(2.*np.pi) - np.log(e2b) - 0.5*(f2a - f2*c)**2/e2b**2
        t1a = -0.5*(f1a - f1)**2/e1a**2
        t1b = -0.5*(f1b - f1*c)**2/e1b**2
        t2a = -0.5*(f2a - f2)**2/e2a**2
        t2b = -0.5*(f2b - f2*c)**2/e2b**2
        return -t1a -t1b -t2a -t2b

    guess = (max(f1a, e1a), max(f2a, e2a), max(f1a/f1b, 0.01))

    res = minimize(nloglike, guess, method='L-BFGS-B', bounds=((0., f1a + 5.*e1a), (0., f2a + 5.*e2a), (0.001, 1000.)), tol=0.1)

    ml = -nloglike(res.x)

    if ml < -0.5*nsigma:
        samecolor = False
    else:
        samecolor = True

    return samecolor, res.x, ml


def find_peaks(image, sigma, peakthre=5., smooth=3):

    ny, nx = image.shape

    peaks = []
    for i in range(ny):
        minrow = max(0, i-smooth)
        maxrow = min(ny, i + smooth)

        for j in range(nx):
            mincol = max(0, j-smooth)
            maxcol = min(nx, j+smooth)

            iloc = image[minrow:maxrow, mincol:maxcol]

            if image[i, j] > peakthre*sigma[i, j]:
                if image[i, j]>= iloc.max():
                    peaks.append((i, j))

    return peaks


def find_objects(image, sigma, peakthre=5., bthre=2., smooth=3):

    ny, nx = image.shape

    peaks = find_peaks(image, sigma, peakthre=peakthre, smooth=smooth)

    index_map = np.zeros((ny, nx), dtype=int)

    count = 0
    for peak in peaks:
        count += 1
        index_map[peak] = count
        queue = [peak]
        while len(queue) > 0:
            pix = queue[0]
            neighbors = []
            if pix[0] > 0:
                neighbors.append((pix[0]-1, pix[1]))
            if pix[0] < ny-1:
                neighbors.append((pix[0]+1, pix[1]))
            if pix[1] > 0:
                neighbors.append((pix[0], pix[1]-1))
            if pix[1] < nx-1:
                neighbors.append((pix[0], pix[1]+1))

            for neighbor in neighbors:
                if image[neighbor] > bthre*sigma[neighbor]:
                    if index_map[neighbor] == 0:
                        index_map[neighbor] = count
                        queue.append(neighbor)

            del queue[0]

    return index_map


def find_objects_twobands(image1, image2, sigma1, sigma2, peakthre=5., bthre=2., cthre=0.2, smooth=3):

    ny, nx = image1.shape

    peaks = find_peaks(image1, sigma1, peakthre=peakthre, smooth=smooth)

    index_map = np.zeros((ny, nx), dtype=int)

    count = 0
    for peak in peaks:
        count += 1
        index_map[peak] = count
        queue = [peak]
        colors = [image2[peak]/image1[peak]]
        med_color = -2.5*np.log10(colors[0])

        while len(queue) > 0:
            pix = queue[0]
            neighbors = []
            if pix[0] > 0:
                neighbors.append((pix[0]-1, pix[1]))
            if pix[0] < ny-1:
                neighbors.append((pix[0]+1, pix[1]))
            if pix[1] > 0:
                neighbors.append((pix[0], pix[1]-1))
            if pix[1] < nx-1:
                neighbors.append((pix[0], pix[1]+1))

            for neighbor in neighbors:
                if image1[neighbor] > bthre*sigma1[neighbor]:
                    fratio = image2[neighbor]/image1[neighbor]
                    if index_map[neighbor] == 0:
                        index_map[neighbor] = count
                        queue.append(neighbor)
                        med_color

            del queue[0]

    return index_map

