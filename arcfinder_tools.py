import numpy as np

def get_arcpos(arcfile):

    f = open(arcfile, 'r')
    lines = f.readlines()
    f.close()

    line = lines[0].split()
    x = int(line[6])
    y = int(line[7])

    return (x, y)
