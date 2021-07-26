import os
import numpy as np


catalog_file = 'yatta_input.cat'
summary_file = 'yatta_output.summary'

rootdir = os.environ.get('YATTADIR')
datadir = os.environ.get('YATTA_DATADIR')
logdir = os.environ.get('YATTA_LOGDIR')
modeldir = os.environ.get('YATTA_MODELDIR')
figdir = os.environ.get('YATTA_FIGDIR')

expected_size = None

if datadir is None:
    datadir = 'data/'

if logdir is None:
    logdir = 'logs/'

if modeldir is None:
    modeldir = 'models/'

if figdir is None:
    figdir = 'figs/'

makeallfigs = False
saveallmodels = False
cleanupdir = True
cutout_data = False

maxarcdist = 30.
minarcdist = 3.
minobjdist = 3.
minarcsize = 20.
maxarcsize = 500.
maxarcdang = 30.
junkstart = 1.3 #1.3
modelmaxdist = 1.3 #1.4
abmin = 1.4
min_aperture = 35.
tooclose = 0.03
se_minap = 25.
color_maxdiff = 2.

# light model parameters
lightfit_reff_max = 30.
lightfit_nser_max = 8.
lightfit_nser_min = 0.5
lightfitrmax = 20.
lightfit_method = 'MCMC'

color_nsigma = 2.

#chi2_thresh = 1000000
#rms_thresh = 10.

source_range = 8.

fitband = 'g'
lightband = 'i'

rgbbands = ('i', 'r', 'g')
rgb_scales = (0.4, 0.6, 1.8)

allbands = list(rgbbands)
if not fitband in allbands:
    allbands.append(fitband)
if not lightband in allbands:
    allbands.append(lightband)

def_config = {'catalog_file': catalog_file, \
              'summary_file': summary_file, \
              'datadir': datadir, \
              'datasourcedir': datadir, \
              'logdir': logdir, \
              'modeldir': modeldir, \
              'figdir': figdir, \
              'lightfitrmax': lightfitrmax, \
              'lightfit_reff_max': lightfit_reff_max, \
              'lightfit_nser_max': lightfit_nser_max, \
              'lightfit_nser_min': lightfit_nser_min, \
              'lightfit_method': lightfit_method, \
              'maxarcdist': maxarcdist, \
              'minarcdist': minarcdist, \
              'maxarcsize': maxarcsize, \
              'minarcsize': minarcsize, \
              'maxarcdang': maxarcdang, \
              'minobjdist': minobjdist, \
              'junkstart': junkstart, \
              'modelmaxdist': modelmaxdist, \
              'abmin': abmin, \
              'min_aperture': min_aperture, \
#              'tooclose': tooclose, \
              'se_minap': se_minap, \
              'color_maxdiff': color_maxdiff, \
              'color_nsigma': color_nsigma, \
              #'chi2_thresh': chi2_thresh, \
              #'rms_thresh': rms_thresh, \
              'source_range': source_range, \
              'fitband': fitband, \
              'lightband': lightband, \
              'rgbbands': rgbbands, \
              'rgb_scales': rgb_scales, \
              'makeallfigs': makeallfigs, \
              'saveallmodels': saveallmodels, \
              'cutout_data': cutout_data, \
              'cleanupdir': cleanupdir, \
              'expected_size': expected_size}

floatpars = ['maxarcdist', 'minarcdist', 'maxarcsize', 'minarcsize', 'maxarcdang', 'minobjdist', 'junkstart', \
             'modelmaxdist', 'abmin', 'min_aperture', 'se_minap', 'color_maxdiff', 'color_nsisgma', 'source_range', \
             'lightfitrmax', 'lightfit_reff_max', 'lightfit_nser_max', 'lightfit_nser_min']

stringpars = ['datadir', 'datasourcedir', 'logdir', 'modeldir', 'figdir', 'fitband', 'lightband', 'rgbbands', 'rgb_scales', 'catalog_file', 'summary_file', 'expected_size', 'lightfit_method']

boolpars = ['makeallfigs', 'saveallmodels', 'cleanupdir', 'cutout_data']

def write_config_file():

    lines = []
    lines.append('# GENERAL PARAMETERS\n')
    lines.append('CATALOG_FILE %s\t# name of input file (or of single object)\n'%def_config['catalog_file'])
    lines.append('SUMMARY_FILE %s\t# name of output file\n'%def_config['summary_file'])
    lines.append('DATADIR %s\t# directory where the data is read from\n'%def_config['datadir'])
    lines.append('LOGDIR %s\t # directory where logs are stored\n'%def_config['logdir'])
    lines.append('MODELDIR %s\t # directory where modeling products are stored\n'%def_config['modeldir'])
    lines.append('FIGDIR %s\t # directory where figures are stored\n'%def_config['figdir'])
    lines.append('FITBAND %s\t# band used for lens modeling\n'%def_config['fitband'])
    lines.append('LIGHTBAND %s\t# band used for lens light and foreground fitting\n'%def_config['lightband'])
    lines.append('RGBBANDS %s, %s, %s\t# bands used for production of rgb images\n'%def_config['rgbbands'])
    lines.append('RGB_SCALES %2.1f, %2.1f, %2.1f\t# scales rgb images\n'%def_config['rgb_scales'])
    lines.append('MAKEALLFIGS %s\t# if YES or True, makes a figure for each candidate\n'%(def_config['makeallfigs']))
    lines.append('SAVEALLMODELS %s\t# if YES or True, saves models of each candidate\n'%(def_config['saveallmodels']))
    lines.append('CLEANUPDIR %s\t# if YES or True, removes intermediate files from working directory\n'%(def_config['cleanupdir']))
    lines.append('CUTOUT_DATA %s\t# if YES or True, uses the HSC pipeline to make cutouts\n'%(def_config['cutout_data']))
    lines.append('EXPECTED_SIZE %s\t# expected image size in number of pixels (sanity check)\n'%(def_config['expected_size']))
    lines.append('\n')
    lines.append('# LENS LIGHT FITTING PARAMETERS\n')
    lines.append('LIGHTFITRMAX %2.1f\t# radius of region used for fitting of the lens light\n'%def_config['lightfitrmax'])
    lines.append('LIGHTFIT_REFF_MAX %2.1f\t# maximum allowed half-light radius\n'%def_config['lightfit_reff_max'])
    lines.append('LIGHTFIT_NSER_MAX %2.1f\t# maximum allowed Sersic index\n'%def_config['lightfit_nser_max'])
    lines.append('LIGHTFIT_NSER_MIN %2.1f\t# minimum allowed Sersic index\n'%def_config['lightfit_nser_min'])
    lines.append('LIGHTFIT_METHOD %s\t# method used for fitting of the lens light (MCMC or minimize)\n'%def_config['lightfit_method'])
    lines.append('# ARC SEARCH PARAMETERS\n')
    lines.append('MAXARCDIST %2.1f\t# maximum search distance from image center\n'%def_config['maxarcdist'])
    lines.append('MINARCDIST %2.1f\t# minimum search distance from image center\n'%def_config['minarcdist'])
    lines.append('MAXARCSIZE %d\t# maximum area of arc footprint\n'%def_config['maxarcsize'])
    lines.append('MINARCSIZE %d\t# minimum area of arc footprint\n'%def_config['minarcsize'])
    lines.append('MAXARCDANG %d\t# maximum angle between arc and tangential curve\n'%def_config['maxarcdang'])
    lines.append('ABMIN %2.1f\t# minimum axis ratio\n'%def_config['abmin'])
    lines.append('SE_MINAP %d\t# minimum angle subtended by arc candidate\n'%def_config['se_minap'])
    lines.append('COLOR_MAXDIFF %2.1f\t# maximum FITBAND-LIGHTBAND color of arc candidate\n'%def_config['color_maxdiff'])
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
    #lines.append('CHI2_THRESH %d\t# maximum reduced chi2 allowed\n'%def_config['chi2_thresh'])

    f = open('default.yatta', 'w')
    f.writelines(lines)
    f.close()

def read_config_file(filename='default.yatta'):

    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    config = def_config.copy()

    for line in lines:
        if line[0] != '#' and len(line) > 1:
            fields = line.split()
            parname = fields[0].lower()
            if parname in config:
                if parname in floatpars:
                    if parname == 'rgb_scales':
                        config[parname] = (float(fields[0]), float(fields[1]), float(fields[2]))
                    else:
                        config[parname] = float(fields[1])

                elif parname in stringpars:
                    if parname == 'rgbbands':
                        rgbbands = []
                        if '#' in line:
                            rgbline = line.split('#')[0].split('RGBBANDS')[1].split(',')
                            for band in rgbline:
                                rgbbands.append(band.strip())
                        config['rgbbands'] = rgbbands
                    elif parname == 'expected_size':
                        try:
                            config['expected_size'] = int(fields[1])
                        except:
                            config['expected_size'] = None
                    else:
                        config[parname] = fields[1]

                elif parname in boolpars:
                    if fields[1] == 'True' or fields[1] == 'YES':
                        config[parname] = True
                    else:
                        config[parname] = False

    allbands = []
    for band in config['rgbbands']:
        allbands.append(band)
    if not config['fitband'] in allbands:
        allbands.append(config['fitband'])
    if not config['lightband'] in allbands:
        allbands.append(config['lightband'])

    config['allbands'] = allbands

    return config

