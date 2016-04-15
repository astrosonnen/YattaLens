import pyfits
import numpy as np


def find_objects(image, sigma, peakthre=5., bthre=2., smooth=3):

    ny, nx = image.shape

    mask = np.zeros((ny, nx))

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
                    mask[i, j] = 1
                    peaks.append((i, j))

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

