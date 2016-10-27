import os
import numpy as np


rootdir = os.environ.get('YATTADIR')
datadir = os.environ.get('YATTA_DATADIR')
logdir = os.environ.get('YATTA_LOGDIR')
modeldir = os.environ.get('YATTA_MODELDIR')
figdir = os.environ.get('YATTA_FIGDIR')

nx_expected = 101
ny_expected = 101

if datadir is None:
    datadir = 'data/'

if logdir is None:
    logdir = 'logs/'

if modeldir is None:
    modeldir = 'models/'

if figdir is None:
    figdir = 'figs/'

makeallfigs = False
maxarcdist = 30.
minarcdist = 3.
minobjdist = 3.
minarcsize = 20.
maxarcsize = 500.
maxarcdang = 30.
junkstart = 1.3 #1.3
modeluntil = 1.4 #1.4
abmin = 1.3
min_aperture = 35.
tooclose = 0.03
se_minap = 25.
gmi_max = 2.
lightfitrmax = 20.

color_nsigma = 2.

chi2_thresh = 1000000
rms_thresh = 10.

source_range = 8.

fitband = 'g'
lightband = 'i'

rgbbands = ('i', 'r', 'g')

allbands = list(rgbbands)
if not fitband in allbands:
    allbands.append(fitband)
if not lightband in allbands:
    allbands.append(lightband)

def_config = {'datadir': datadir, \
              'logdir': logdir, \
              'modeldir': modeldir, \
              'figdir': figdir, \
              'maxarcdist': maxarcdist, \
              'minarcdist': minarcdist, \
              'maxarcsize': maxarcsize, \
              'minarcsize': minarcsize, \
              'maxarcdang': maxarcdang, \
              'minobjdist': minobjdist, \
              'junkstart': junkstart, \
              'modelmaxdist': modeluntil, \
              'abmin': abmin, \
              'min_aperture': min_aperture, \
#              'tooclose': tooclose, \
              'se_minap': se_minap, \
              'gmi_max': gmi_max, \
              'color_nsigma': color_nsigma, \
              'chi2_thresh': chi2_thresh, \
              'rms_thresh': rms_thresh, \
              'source_range': source_range, \
              'fitband': fitband, \
              'lightband': lightband, \
              'rgbbands': rgbbands}

def write_config_file():

    lines = []
    lines.append('# GENERAL PARAMETERS\n')
    lines.append('DATADIR ./\t# directory where the data is read from\n')
    lines.append('LOGDIR ./logs/\t # directory where logs are stored\n')
    lines.append('MODELDIR ./models/\t # directory where modeling products are stored\n')
    lines.append('FIGDIR ./figs/\t # directory where figures are stored\n')
    lines.append('CANDIDATE_CLASSIFICATION YES # if YES, tests likelihood of object being a lens\n')
    lines.append('FITBAND %s\t# band used for lens modeling\n'%def_config['fitband'])
    lines.append('LIGHTBAND %s\t# band used for lens light and foreground fitting\n'%def_config['lightband'])
    lines.append('RGBBANDS i, r, g\t# bands used for production of rgb images\n')
    lines.append('\n')
    lines.append('# ARC SEARCH PARAMETERS\n')
    lines.append('MAXARCDIST %2.1f\t# maximum search distance from image center\n'%def_config['maxarcdist'])
    lines.append('MINARCDIST %2.1f\t# minimum search distance from image center\n'%def_config['minarcdist'])
    lines.append('MAXARCSIZE %d\t# maximum area of arc footprint\n'%def_config['maxarcsize'])
    lines.append('MINARCSIZE %d\t# minimum area of arc footprint\n'%def_config['minarcsize'])
    lines.append('MAXARCDANG %d\t# maximum angle between arc and tangential curve\n'%def_config['maxarcsize'])
    lines.append('ABMIN %2.1f\t# minimum axis ratio\n'%def_config['abmin'])
    lines.append('SE_MINAP %d\t# minimum angle subtended by arc candidate\n'%def_config['se_minap'])
    lines.append('GMI_MAX %2.1f\t# maximum g-i color of arc candidate\n'%def_config['gmi_max'])
    lines.append('COLOR_NSIGMA %2.1f\t# number of sigmas required for color consistency\n'%def_config['color_nsigma'])
    lines.append('\n')
    lines.append('# FOREGROUND OBJECT SEARCH PARAMETERS\n')
    lines.append('MINOBJDIST %2.1f\t# minimum search distance from image center\n'%def_config['minobjdist'])
    lines.append('JUNKSTART %2.1f\t#distance (in units of distance of the farthest arc candidate) beyond which objects are labeled as foregrounds\n'%def_config['junkstart'])
    lines.append('MODELMAXDIST %2.1f\t# distance (in units of distance of the farthest arc candidate) beyond which foregrounds are not modeled\n'%def_config['modelmaxdist'])
    lines.append('\n')
    lines.append('# LENS MODEL PARAMETERS\n')
    lines.append('SOURCE_RANGE %2.1f\t'%def_config['source_range'])
    lines.append('\n')
    lines.append('# CANDIDATE CLASSIFICATION PARAMETERS\n')
    lines.append('MIN_APERTURE %d\t# minimum angle subtended by the lensed source\n'%def_config['min_aperture'])
    lines.append('CHI2_THRESH %d\t# maximum reduced chi2 allowed\n'%def_config['chi2_thresh'])

    f = open('default.yatta', 'w')
    f.writelines(lines)
    f.close()


