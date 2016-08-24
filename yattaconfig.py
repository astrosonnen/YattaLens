import os

datadir = os.environ.get('YATTA_DATADIR')
logdir = os.environ.get('YATTA_LOGDIR')
modeldir = os.environ.get('YATTA_MODELDIR')

if datadir is None:
    datadir = 'data/'

if logdir is None:
    logdir = 'logs/'

if modeldir is None:
    modeldir = 'models/'

maxarcdist = 30.
minarcdist = 3.
minarcsize = 20.
maxarcsize = 500.
maxarcdang = 30.
junkstart = 1.3
modeluntil = 1.4
abmin = 1.4
min_aperture = 35.
tooclose = 0.03
se_minap = 25.
gmi_max = 2.

chi2_thresh = 200.
rms_thresh = 10.

source_range = 8.

fitband = ('g')
lightband = 'i'

rgbbands = ('i', 'r', 'g')

