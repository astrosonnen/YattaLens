#======================================================================================================================
# IMPORTING MODULES
import numpy
import pylab
import pyfits
import pymc
import convolve,SBModels
from pylens import pylens,MassModels
from photofit import indexTricks as iT
import cPickle
from math import log10,ceil
#------------ the following ones are for sampler and optimizer
from photofit.SampleOpt import AMAOpt,Sampler
#------------ the following ones are for "getModel"
from scipy import optimize

#======================================================================================================================
def wrapper(sys_para, instr_para, ep, psf_name, Nsample, Ncov, burnin, flag):
    """
    This subroutine modulizes the main function "example4xiaolei.py" of the pylens software package
    
    --- INPUT ---
    sys_para    :   an ndim=1 array containing parameters for the lens+source systems
    instr_para  :   an ndim=1 array containing parameters for the observational instruments (HST, LSST, etc.)
    psf_name    :   a string leading to the path+name of the psf fits file
    Nsample     :   an int giving the sample size of MCMC fitting
    burnin      :   - an int (>=1) which is the number of points to be burnt        *or*
                    - a float (<1) which is the portion of Nsample to be burnt
    flag        :   a boolean deciding whether to run MCMC ('True') ot not ('False', in this case showing simulated images instead)

    === OUTPUT ===
    rslt = sampler.result()     :   an n=3 tuple, if flag=False, it is (None,None,None), otherwise
                            [0] - logP   : ndarray, shape=(Nsample,) , recording the log(posterior)
                            [1] - trace  : ndarray, shape=(Nsample, num of param in MCMC), recording param values
                            [2] - result : dict type quantity, basically the same with [1], but with header info included
    """
    assert (ep > 1), 'You input erroneous exposure time = %s sec' + str(ep)
    zp = float(instr_para[2])       # zero point
    pix_num = float(instr_para[5])     # pixel number

    #--------------------------------------------------------------------------------------
    # Create surface brightness objects that define the model
    lensGal = models.Sersic('lens',{'x':sys_para[1],'y':sys_para[2],'re':sys_para[3],'q':sys_para[4],'pa':sys_para[5],'n':sys_para[6]})
    mag_lens = sys_para[7]

    lensMag = lensGal.Mag(zp)
    lensGal.amp = 10.**(-0.4*(mag_lens-lensMag))

    src = SBModels.Sersic('src',{'x':sys_para[8],'y':sys_para[9],'re':sys_para[10],'q':sys_para[11],'pa':sys_para[12],'n':sys_para[13]})
    mag_src = sys_para[14]
    srcMag = src.Mag(zp)
    src.amp = 10.**(-0.4*(mag_src-srcMag))

    #--------------------------------------------------------------------------------------
    # Create mass (or lensing) objects
    lensMass = MassModels.PowerLaw('lens',{'x':sys_para[15],'y':sys_para[16],'b':sys_para[17],'q':sys_para[18],'pa':sys_para[19],'eta':sys_para[20]})
#    shear = MassModels.ExtShear('shear',{'x':sys_para[34],'y':sys_para[35],'b':sys_para[36],'pa':sys_para[37]})
#    lenses = [lensMass,shear]
    lenses = [lensMass]

    #--------------------------------------------------------------------------------------
    #create the coordinate grids that the image will be evaluated on
    y,x = iT.coords((pix_num,pix_num))

    #--------------------------------------------------------------------------------------
    # create a PSF   
    psf = pyfits.open(psf_name)[0].data
    psf /= psf.sum()

    #--------------------------------------------------------------------------------------
    # Form the image and convolve (convolution returns the image and FFT'd PSF)
    b = lensGal.pixeval(x,y)
    img,psf_fftd = convolve.convolve(lensGal.pixeval(x,y),psf)

    #--------------------------------------------------------------------------------------
    xl,yl = pylens.getDeflections(lenses,[x,y])
    img += convolve.convolve(src.pixeval(xl,yl),psf_fftd,False)[0]
    counts = numpy.array(img)

    #--------------------------------------------------------------------------------------    
    # a switch to turn on (flag=True) MCMC or not
    if flag == False:
        # Plotting img
        pylab.imshow(img,origin='lower',interpolation='nearest')
        pylab.colorbar()
        pylab.savefig('TMT_10000s_135949_img_withlens.png',dpi=200)
        # pylab.ion()
        pylab.show()
        pyfits.PrimaryHDU(img).writeto('TMT_10000s_135949_img_withlens.fits',clobber=True)
        rslt = (None,None,None)

    else:
    #--------------------------------------------------------------------------------------
    # Now if you wanted to _fit_ an image you'd do mostly the same thing, but
    #  use variables for the parameters instead of constants

        # Define the variables for the lens galaxy light
        GX = pymc.Uniform('gx',sys_para[1]-sys_para[1]/10.,sys_para[1]+sys_para[1]/10.,value=sys_para[1])
        GY = pymc.Uniform('gy',sys_para[2]-sys_para[2]/10.,sys_para[2]+sys_para[2]/10.,value=sys_para[2])
        GR = pymc.Uniform('gr',sys_para[3]-0.5*sys_para[3],sys_para[3]+0.5*sys_para[3],value=sys_para[3])
        GQ = pymc.Uniform('gq',0.1,1.,value=sys_para[4])
        GP = pymc.Uniform('gp',-180.,180.,value=sys_para[5])
        GN = pymc.Uniform('gn',0.5,8.,value=sys_para[6])

        # Variables for the source light; we'll start with a spherical n=1 object
        SX = pymc.Uniform('sx',sys_para[8]-sys_para[8]/10.,sys_para[8]+sys_para[8]/10.,value=sys_para[8])
        SY = pymc.Uniform('sy',sys_para[9]-sys_para[9]/10.,sys_para[9]+sys_para[9]/10.,value=sys_para[9])
        SR = pymc.Uniform('sr',sys_para[10]-0.5*sys_para[10],sys_para[10]+0.5*sys_para[10],value=sys_para[10])
        SQ = pymc.Uniform('sq',0.1,1.,value=sys_para[11])
        SP = pymc.Uniform('sp',-180.,180.,value=sys_para[12])
        SN = pymc.Uniform('sn',0.5,8.,value=sys_para[13])

        # Variables for the mass models; we'll have a prior tying the mass position
        #   to the light but we'll leave the ellipticity free here
        LX = pymc.Normal('lx',GX,1./3**2,value=GX.value) # Gaussian with sigma = 3
        LY = pymc.Normal('ly',GY,1./3**2,value=GY.value) # Use precision, not sigma...
        LB = pymc.Uniform('lb',sys_para[17]-0.5*sys_para[17],sys_para[17]+0.5*sys_para[17],value=sys_para[17])
        LQ = pymc.Uniform('lq',0.1,1.,value=sys_para[18])
        LP = pymc.Uniform('lp',-180.,180.,value=sys_para[19])
        LE = pymc.Uniform('le',0.7,1.3,value=sys_para[20])

#        XB = pymc.Uniform('shear',0.,0.,value=sys_para[36])
#        XP = pymc.Uniform('shear pa',0.,0.,value=sys_para[37])


        # Now create the models as before, but with variables instead of constants!
        lensGal = SBModels.Sersic('lens',{'x':GX,'y':GY,'re':GR,'q':GQ,'pa':GP,'n':GN})
        src = SBModels.Sersic('src',{'x':SX,'y':SY,'re':SR,'q':SQ,'pa':SP,'n':SN})

        lens = MassModels.PowerLaw('lens',{'x':LX,'y':LY,'b':LB,'q':LQ,'pa':LP,'eta':LE})
        # The shear origin will be on the mass
#        shear = MassModels.ExtShear('shear',{'x':LX,'y':LY,'b':XB,'pa':XP})
#        lenses = [lens,shear]
        lenses = [lens]

        # Create a list for the parameters, and define starting proposal steps for the optimization
        pars = [GX,GY,GR,GQ,GP,GN]
        cov = [0.05,0.05,0.1,0.01,3.,0.1]
        pars = pars+[SX,SY,SR,SQ,SP,SN]
        cov = cov+[0.1,0.1,0.1,0.01,3.,0.1]
#        pars = pars+[LX,LY,LB,LQ,LP,LE,XB,XP]
#        cov = cov+[0.01,0.01,0.05,0.01,3.,0.01,0.005,3.]
        pars = pars+[LX,LY,LB,LQ,LP,LE]
        cov = cov+[0.01,0.01,0.05,0.01,3.,0.01]

        # Finally define the optimization function; this evaluates the model and the
        #   current parameter values and returns the residual
        @pymc.observed
        def logP(value=0,tmpXXX=pars): # This is a non-standard use of pymc....
            chi = getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, agn3, agn4, psf_fftd)[0]
#            chi = getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, psf_fftd)[0]
            return -0.5*chi**2

        # Now optimize twice to locate the global maximum of llh
#        for i in range(2):
#            print '-------------------------- optimizer loop i=%s' % str(i+1)
#            sampler = AMAOpt(pars,[logP],[],cov=numpy.array(cov))
#            sampler.sample(Nsample)

        # Below switch from optimizer to sampler
        #------ setting up burnin number
        assert (burnin > 0), 'You input negative burnin value = %s' + str(burnin)
        if burnin < 1:
            burnin = int(Nsample * burnin)
#        print '----------------- 1st step: obtain optm proposal mat'
        sampler = Sampler(pars,[logP],[])
        sampler.setCov(Ncov*numpy.array(cov))
        sampler.sample(Nsample)
        logp,trace,result = sampler.result()

        g = open('result_lp1.cpkl','wb')
        cPickle.dump(sampler.result(),g,2)
        g.close()

#        pylab.figure(1)
#        pylab.ioff()
#        pylab.imshow(numpy.corrcoef(trace[burnin:].T),origin='lower',interpolation='nearest')
#        pylab.colorbar()
#        pylab.savefig('TMT_360s_30000_burnin0.2_1lp_pro_Ncov0.09.png',dpi=200)
#        pylab.show()

#        pylab.figure(2)
#        pylab.ioff()
#        pylab.plot(result['le'])
#        pylab.savefig('TMT_360s_30000_burnin0.2_1lp_le_Ncov0.09.png',dpi=200)
#        pylab.show()

        #------ updating optm proposal mat        
        cov_optm = numpy.cov(trace[burnin:].T)

#        print '================= 2nd step: begin to do true MCMC sampling!'
        #------ updating the start values       ->      no need to do this! start values are automatically updated!
        #for i in range(len(pars)):
        #    pars[i].value = trace[-1][i]
        #------ sample Nsample times to obtain the final MCMC data
        sampler = Sampler(pars,[logP],[])
        sampler.setCov(numpy.array(cov_optm))
        sampler.sample(Nsample)
        #------ get results from sampler -- "trace" contains samples from the posterior, calc cov/mean/median on it!
        rslt = sampler.result()

#        pylab.figure(3)
#        pylab.ioff()
#        pylab.plot(rslt[2]['le'])
#        pylab.savefig('TMT_360s_30000_burnin0.2_2lp_Ncov0.09.png',dpi=200)

    return rslt
#======================================================================================================================
def getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, agn3, agn4, psf_fftd):
#def getModel(img, noise, lensGal, src, lenses, x, y, agn1, agn2, psf_fftd):
    """
    This subroutine calculates the value of chi
    cut from the original built-in subroutine at pylens main function
    """
#    print '#------------ enter getModel'
# This evaluates the model at the given parameters and finds the best amplitudes for the surface brightness components
    I = img.flatten()
    S = noise.flatten()
# We need to update the parameters of the objects to the values proposed by the optimizer
    lensGal.setPars()
    src.setPars()
    for l in lenses:
#    for l in lens:
        l.setPars()

    xl,yl = pylens.getDeflections(lenses,[x,y])
#    xl,yl = pylens.getDeflections(lens,[x,y])

    model = numpy.empty((img.size,6))
    model[:,0] = convolve.convolve(lensGal.pixeval(x,y),psf_fftd,False)[0].ravel()/S
    model[:,1] = convolve.convolve(src.pixeval(xl,yl),psf_fftd,False)[0].ravel()/S

    amps,chi = optimize.nnls(model,I/S)
#    print '#============ exit getModel'
    return chi,amps*(model.T*S).T


#======================================================================================================================
def run(sys_para, instr_para, ep_para, psf_name, outputFileName=None, Nsample=2000, Ncov=0.01, burnin=500, flag=True):
    """
    This subroutine conducts simulation and MCMC fitting iteratively
    *NOTE: here sys_para.ndim can be any int value besides 1!
    
    === OUTPUT ===
        a file in cPickle(protocol=2) format with the name given by "outputFileName" recording all MCMC fitting results,
        after reading in, it manifests as a tuple with n=sys_para.ndim (including 1!), 
        each element is an n=3 tuple reflecting the MCMC result given by "sampler.result()" in _wrapper_

        *NOTE: the current concatenation method of tuples is just a tempory trick, 
               suggestions on more sophisticated method greatly welcome!
    """
    mcmcrslt = ()
    if sys_para.ndim == 1:
        mcmcrslt_new = wrapper(sys_para, instr_para, ep_para, psf_name, Nsample,  Ncov, burnin, flag)
        mcmcrslt = mcmcrslt + (mcmcrslt_new,)
        mcmcrslt_new = None

    else:
        Nobj=len(sys_para)
        for i in xrange(Nobj):
            print '#----------------- begin running _wrapper_ on No. %s lens system' % str(i+1)
            mcmcrslt_new=wrapper(sys_para[i,:], instr_para, ep_para, psf_name, Nsample, Ncov, burnin, flag)
            print 'Simulating and MCMC fitting No. '+str(i+1)+' lens system completed!'
            mcmcrslt = mcmcrslt + (mcmcrslt_new,)
            mcmcrslt_new = None

    # dump the tuple (containing all results) into a pickle in the most compact format (protocol=2)
    if outputFileName != None and flag == True:
        f = open(outputFileName,'wb')
        cPickle.dump(mcmcrslt,f,2)
        f.close()

       
