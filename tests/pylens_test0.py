#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from xlmengtools import lens_fitting_4agn
import numpy
import time
t = time.time()

workdir=    '/data1/homedirs/xlmeng/workplace/135949+553550/'
sys_name=   workdir+'HST/src_lens_HST_para.txt'
instr_name= workdir+'teleParam.txt'
#ep_name=    workdir+'Fitting/ep_para.txt'
psf_name=   workdir+'PSF/HST_F814W.fits'

sys_allpara=numpy.genfromtxt(sys_name,dtype=float,comments='#')
sys_chosen=sys_allpara[0:1,:]

# get all data (including telescope names) in string format 
instr_allpara  = numpy.genfromtxt(instr_name,dtype=str,comments='#')
instr_chosen= instr_allpara[0,:]
#<<<140826>>>NOTE: currently you can only calc instrument one by one, but no need to specify its name by yourself

ep = [1000.0, 3000.0, 9000.0]
ep_chosen = ep[0:1]

# number of chosen exposure times
Nep = numpy.array(ep_chosen).size

# number of iterations
Niter = 20

# MCMC sample parameters
Nsample = 30000
burnin  = 0.2

# times of the cov
Ncov = 0.01

# execute pylens
print '============== Start doing telescope %s at filter %s' % (instr_chosen[0],instr_chosen[1])
for i in xrange(Nep):
    print '---------- Start doing exposure time = %s sec' % str(ep_chosen[i])
    for j in xrange(Niter):
        outputFileName = instr_chosen[0]+'_'+instr_chosen[1]+'_'+str(int(ep_chosen[i]))+'s_iter'+str(j+1)+'.cpkl'
        print 'Iteration No. ', j+1, 'output file name =', outputFileName
        lens_fitting_4agn.run(sys_chosen, instr_chosen, ep_chosen[i], psf_name, outputFileName, Nsample, Ncov, burnin, flag=True)

# calc the time elapsed
elapsed = time.time() - t
print 'Elapsed time running the code: %s sec' % str(elapsed)

