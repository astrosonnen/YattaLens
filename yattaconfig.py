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

fitband = ['g']
lightband = ['i']
