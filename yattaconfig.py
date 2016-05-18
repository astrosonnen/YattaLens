import os

datadir = os.environ.get('YATTA_DATADIR')
psfdir = os.environ.get('YATTA_PSFDIR')
logdir = os.environ.get('YATTA_LOGDIR')
modeldir = os.environ.get('YATTA_MODELDIR')

if datadir is None:
    datadir = 'cutout_dir/'

if psfdir is None:
    psfdir = datadir

if logdir is None:
    logdir = 'logs/'

if modeldir is None:
    modeldir = 'models/'

