

def getPSFMatrix(psf,imshape,mask=None):
    """
    Create a PSF matrix given the PSF model and image dimensions
    """
    import numpy
    from scipy.sparse import coo_matrix

    imsize = imshape[0]*imshape[1]
    tmp = numpy.zeros(imshape)
    dx = (imshape[1]-psf.shape[1])/2
    dy = (imshape[0]-psf.shape[0])/2
    tmp[dy:-dy,dx:-dx] = psf.copy()

    cvals = numpy.where(tmp.flatten()!=0)[0]
    pvals = tmp.ravel()[cvals]

    row = numpy.arange(imsize).repeat(cvals.size)
    col = numpy.tile(cvals,imsize)+row
    col -= imsize/2-imshape[0]/2
    pvals = numpy.tile(pvals,imsize)

    good = (col>0)&(col<imsize)
    col = col[good]
    row = row[good]
    pvals = pvals[good]

    pmat = coo_matrix((pvals,(col,row)),shape=(imsize,imsize))
    if mask is not None:
        npnts = mask.sum()
        c = numpy.arange(imsize)[mask.ravel()]
        r = numpy.arange(npnts)
        smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
        pmat = smat.T*(pmat*smat)
    return pmat


def maskPSFMatrix(pmat,mask):
    import numpy
    from scipy.sparse import coo_matrix

    imsize = pmat.shape[0]
    npnts = mask.sum()
    c = numpy.arange(imsize)[mask.ravel()]
    r = numpy.arange(npnts)
    smat = coo_matrix((numpy.ones(npnts),(c,r)),shape=(imsize,npnts))
    return smat.T*(pmat*smat)


def getRegularizationMatrix(srcxaxis,srcyaxis,mode="curvature"):
    import numpy
    
    if mode=="zeroth":
        return identity(srcxaxis.size*srcyaxis.size)

    else: from scipy.sparse import diags,csc_matrix,lil_matrix

    if mode=="gradient":
        mat = diags([-2,-2,8,-2,-2],[-srcxaxis.size,-1,0,1,srcxaxis.size],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size))
        mat=lil_matrix(mat)

        #glitches are at left and right edges
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        leftedges=allcols[allcols%srcxaxis.size==0]
        rightedges=allcols[allcols%srcxaxis.size==srcxaxis.size-1]
        for el in leftedges:
            mat[el,el-1]=0
        for el in rightedges:
            if el != allcols.max():
                mat[el,el+1]=0

    elif mode=="curvatureOLD":
        mat=diags([2,2,-8,-8,24,-8,-8,2,2],[-2*srcxaxis.size,-2,-srcxaxis.size,-1,0,1,srcxaxis.size,2*srcxaxis.size,2],shape=(srcxaxis.size*srcyaxis.size,srcxaxis.size*srcyaxis.size)) 
        mat=lil_matrix(mat)
        
        #glitches are at left and right edges
        allcols=numpy.arange(srcxaxis.size*srcyaxis.size)
        leftedges=allcols[allcols%srcxaxis.size==0]
        rightedges=allcols[allcols%srcxaxis.size==srcxaxis.size-1]
        leftedgesinone=allcols[allcols%srcxaxis.size==1]
        rightedgesinone=allcols[allcols%srcxaxis.size==srcxaxis.size-2]

        for el in leftedges:
            mat[el,el-1]=0
            mat[el,el-2]=0
        for el in rightedges:
            if el != allcols.max():
                mat[el,el+1]=0
                mat[el,el+2]=0
        for el in leftedgesinone:
            mat[el,el-2]=0
        for el in rightedgesinone:
            if el != allcols.max()-1:
                mat[el,el+2]=0

    elif mode=="curvature":
        I,J=srcxaxis.size,srcyaxis.size
        matrix=lil_matrix((I*J,I*J))
        for i in range(I-2):
            for j in range(J):
                ij=i+j*J
                i1j=ij+1
                i2j=ij+2
                matrix[ij,ij]+=1.
                matrix[i1j,i1j]+=4
                matrix[i2j,i2j]+=1
                matrix[ij,i2j]+=1
                matrix[i2j,ij]+=1
                matrix[ij,i1j]-=2
                matrix[i1j,ij]-=2
                matrix[i1j,i2j]-=2
                matrix[i2j,i1j]-=2
        for i in range(I):
            for j in range(J-2):
                ij=i+j*J
                ij1=ij+J
                ij2=ij+2*J
                matrix[ij,ij]+=1
                matrix[ij1,ij1]+=4
                matrix[ij2,ij2]+=1
                matrix[ij,ij2]+=1
                matrix[ij2,ij]+=1
                matrix[ij,ij1]-=2
                matrix[ij1,ij]-=2
                matrix[ij1,ij2]-=2
                matrix[ij2,ij1]-=2
        for i in range(I):
            iJ_1=i+(J-2)*J
            iJ=i+(J-1)*J
            matrix[iJ_1,iJ_1]+=1
            matrix[iJ,iJ]+=1
            matrix[iJ,iJ_1]-=1
            matrix[iJ_1,iJ]-=1
        for j in range(J):
            I_1j=(I-2)+j*J
            Ij=(I-1)+j*J
            matrix[I_1j,I_1j]+=1
            matrix[Ij,Ij]+=1
            matrix[Ij,I_1j]-=1
            matrix[I_1j,Ij]-=1
        for i in range(I):
            iJ=i+(J-1)*J
            matrix[iJ,iJ]+=1
        for j in range(J):
            Ij=(I-1)+j*J
            matrix[Ij,Ij]+=1
        mat=matrix
    return mat.tocsc()


def getModel(img,var,lmat,pmat,cmat,rmat,reg,niter=10,onlyRes=False):
    from scikits.sparse.cholmod import cholesky
    import numpy

    omat = pmat*lmat
    rhs = omat.T*(img/var)

    B = omat.T*cmat*omat

    res = 0.
    regs = [reg]
    lhs = B+regs[-1]*rmat

    F = cholesky(lhs)
    fit = F(rhs)
    for i in range(niter):
        res = fit.dot(rmat*fit)

        delta = reg*1e3
        lhs2 = B+(reg+delta)*rmat

        T = (2./delta)*(numpy.log(F.cholesky(lhs2).L().diagonal()).sum()-numpy.log(F.L().diagonal()).sum())
        reg = (omat.shape[0]-T*reg)/res
        if abs(reg-regs[-1])/reg<0.005:
            break
        regs.append(reg)
        lhs = B+regs[-1]*rmat
        F = F.cholesky(lhs)
        fit = F(rhs)
    print reg,regs
    res = -0.5*res*regs[-1] + -0.5*((omat*fit-img)**2/var).sum()
    if onlyRes:
        return res,reg
    model = (omat*fit)
    return res,reg,fit,model


def PixelNumber(x0,y0,xsrcaxes,ysrcaxes,mode='NearestCentre'):
    import numpy
    srcpixscale=xsrcaxes[1]-xsrcaxes[0]
    if mode=='NearestCentre':
        xpixelnumber=(numpy.floor(((x0-xsrcaxes[0])/srcpixscale)+0.5))
        ypixelnumber=(numpy.floor(((y0-ysrcaxes[0])/srcpixscale)+0.5))

    if mode=='NearestBottomLeft':
        xpixelnumber=(numpy.floor(((x0-xsrcaxes[0])/srcpixscale)))
        ypixelnumber=(numpy.floor(((y0-ysrcaxes[0])/srcpixscale)))

    pixelnumber=ypixelnumber*len(xsrcaxes)+xpixelnumber

    pixelnumber[xpixelnumber<0]=-1
    pixelnumber[ypixelnumber<0]=-1
    pixelnumber[xpixelnumber>=len(xsrcaxes)]=-1
    pixelnumber[ypixelnumber>=len(ysrcaxes)]=-1

    if mode=='NearestBottomLeft': 
        #we want the point to be within the grid
        pixelnumber[xpixelnumber==len(xsrcaxes)-1]=-1
        pixelnumber[ypixelnumber==len(ysrcaxes)-1]=-1

    return pixelnumber


def getLensMatrixBilinear(lenses,x,y,srcx,srcy,srcxaxis,srcyaxis,imsize):
    """
    Bilinear lensing matrix
    """
    import numpy
    from scipy.sparse import coo_matrix

    xin = x.copy()
    yin = y.copy()

    # pylens.getDeflections() would do this too....
    for lens in lenses:
        lens.setPars()
        xmap,ymap = lens.deflections(x,y)
        xin -= xmap
        yin -= ymap

    scale = srcxaxis[1]-srcxaxis[0]
    spix = PixelNumber(xin,yin,srcxaxis,srcyaxis,mode='NearestBottomLeft')
    c = spix>-1
    row = numpy.arange(spix.size)[c]
    xin,yin = xin[c],yin[c]
    spix = spix[c].astype(numpy.int32)

    # Create row, col, value arrays
    r = numpy.empty(row.size*4)
    c = numpy.empty(row.size*4)
    w = numpy.empty(row.size*4)

    # These are the lower-right pixel weights
    r[:row.size] = row
    c[:row.size] = spix
    w[:row.size] = (1.-numpy.abs(xin-srcx[spix])/scale)*(1.-numpy.abs(yin-srcy[spix])/scale)

    # Now do the lower-left, upper-left, and upper-right pixels
    a = [1,srcxaxis.size,-1]
    for i in range(1,4):
        spix += a[i-1]
        r[i*row.size:(i+1)*row.size] = row
        c[i*row.size:(i+1)*row.size] = spix
        w[i*row.size:(i+1)*row.size] = (1.-numpy.abs(xin-srcx[spix])/scale)*(1.-numpy.abs(yin-srcy[spix])/scale)

    return coo_matrix((w,(r,c)),shape=(x.size,srcx.size))
