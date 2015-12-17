      program bah
      end

      subroutine sersicdeflections(x,y,amp,r,eta,q,m,dr,k,xout,yout,n)
!f2py double precision dimension(n),intent(in) :: x,y
!f2py double precision deminsion(n),intent(out) :: xout,yout
!f2py integer intent(hide),depend(x) :: n=len(x)
!f2py integer intent(in), optional :: m = 31
!f2py double precision intent(in), optional :: dr = 5.
!f2py integer intent(in), optional :: k = 3
      double precision x(n),y(n),xout(n),yout(n)
      double precision xintegrand(m),yintegrand(m),uvar(m)
      double precision u,u2
      double precision amp,eta,q,x2,y2,q2,u2q2
      double precision lim,delta,w2
      double precision w(m)
      integer iopt,m,k,nest,lwrk,ier,n0
      double precision t(m+k+1),c(m+k+1),wrk(m*(k+1)+(m+k+1)*(7+3*k))
      double precision xb,xe,s,fp,d
      integer iwrk(m+k+1)
      double precision xy,r2,dr,dr2,sq2
      double precision serK,norm,ie,r

      nest = m+k+1
      lwrk = m*(k+1)+nest*(7+3*k)
      iopt = 0
      s = 0.
      do i=1,m
        w(i) = 1.
        uvar(i) = 0.
        xintegrand(i) = 0.
        yintegrand(i) = 0.
      enddo

      q5 = dsqrt(q)
      q2 = q*q
      d = (m-1)/dr
      sq2 = 1.-q2

      serK = 2.*eta-1./3+4./(405.*eta)+46/(25515.*eta**2)
      norm = amp
      serK = serK*-1.
      ie = 1./eta
      do i=1,n
        x2 = x(i)*x(i)
        y2 = y(i)*y(i)
        lim = dlog10(dsqrt(x2+y2/q2))
c        lim = dsqrt(x2+y2/q2)
        xy = x2*y2
        r2 = x2+y2
        dr2 = y2-x2
        do j=1,m
          u = 10.**(lim-dr+(j-1)/d)
c          u = lim*(j-0.9999)/m

          u2 = u*u
          u2q2 = u2*sq2
          delta = dsqrt((u2q2+dr2)*(u2q2+dr2)+4.*xy)
          w2 = (delta+r2+u2q2)/(delta+r2-u2q2)
          xintegrand(j) = dexp(serK*(u/r)**ie)*u*dsqrt(w2)/(x2+y2*w2*w2)
          yintegrand(j) = w2*xintegrand(j)
          uvar(j) = u
        enddo
        xb = uvar(1)
        xe = uvar(m)

        call curfit(iopt,m,uvar,xintegrand,w,xb,xe,k,s,nest,n0,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
        xout(i) = 2*norm*q*x(i)*splint(t,n0,c,k,xb,xe,wrk)
        call curfit(iopt,m,uvar,yintegrand,w,xb,xe,k,s,nest,n0,t,c,fp,
     *             wrk,lwrk,iwrk,ier)
        yout(i) = 2*norm*q*y(i)*splint(t,n0,c,k,xb,xe,wrk)
      enddo
      end

