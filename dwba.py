import numpy as np
import integrals
import numba

def dwba_tilted_qx(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):

    r1,t1,r2,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    tf = t[:,0]
    t1t2 = t1*t2
    t1r2 = t1*r2
    r1t2 = r1*t2
    r1r2 = r1*r2
    
    t1t2c = np.conj(t1t2)
    t1r2c = np.conj(t1r2)
    r1t2c = np.conj(r1t2)
    r1r2c = np.conj(r1r2)
    
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += ((n[m]**2-n[m+1]**2)*np.conj(n[l]**2-n[l+1]**2)*(
                        (t1t2[l]*(t1t2c[m])+r1r2[l]*(r1r2c[m]))
                       +(t1t2[l]*(r1r2c[m])+r1r2[l]*(t1t2c[m]))
                       +(t1r2[l]*(t1t2c[m])+r1t2[l]*(r1r2c[m]))
                       +(r1t2[l]*(t1t2c[m])+t1r2[l]*(r1r2c[m]))
                       +(t1t2[l]*(t1r2c[m])+r1r2[l]*(r1t2c[m]))
                       +(r1r2[l]*(t1r2c[m])+t1t2[l]*(r1t2c[m]))
                       +(t1r2[l]*(t1r2c[m])+r1t2[l]*(r1t2c[m]))
                       +(r1t2[l]*(t1r2c[m])+t1r2[l]*(r1t2c[m]))
                    )*C_perp_qx(l,m,tf,xi_perp,qx)*np.exp(-1j*qx*np.tan(np.radians(beta))*z(l,m,tf)))
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum*C_lateral(qx, sigma, xi_lat, H)



@numba.jit(nopython=True, cache=False)
def dwba_tilted_qx_fast(res, wl, angle_in, xi_lat, xi_perp, hurst, sigma, beta):
    r1, t1, r2, t2, qx, qz, t, n = res
    r1,t1,r2,t2 =r1[:-1,:],t1[:-1,:],r2[:-1,:],t2[:-1,:]
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    tf = t[:,0]
    
    z_mat = np.empty((len(t1), len(t1)), dtype=np.float64)
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            if l > m:
                z_mat[l,m] = np.sum(tf[m:(l-1)])
            elif l < m:
                z_mat[l,m] = -np.sum(tf[l:(m-1)])
            else:
                z_mat[l,m] = tf[l]

    Cl = integrals.C_lateral(qx, sigma, xi_lat, H)
    return dwba_tilted_qx_fast_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)

@numba.jit(nopython=True, cache=False)
def dwba_tilted_qx_fast_mhe(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
    r1,t1,r2,t2 = res
    r1,t1,r2,t2 =r1[:-1,:],t1[:-1,:],r2[:-1,:],t2[:-1,:]
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    tf = t[:,0]
    
    z_mat = np.empty((len(t1), len(t1)), dtype=np.float64)
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            if l > m:
                z_mat[l,m] = np.sum(tf[m:(l-1)])
            elif l < m:
                z_mat[l,m] = -np.sum(tf[l:(m-1)])
            else:
                z_mat[l,m] = tf[l]

    Cl = np.ones(len(Cl))
    return dwba_tilted_qx_fast_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)

@numba.jit(nopython=True, cache=False)
def dwba_tilted_qx_fast_mhe_sk(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
    r1,t1,r2,t2 = res
    r1,t1,r2,t2 =r1[:-1,:],t1[:-1,:],r2[:-1,:],t2[:-1,:]
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    tf = t[:,0]
    
    z_mat = np.empty((len(t1), len(t1)), dtype=np.float64)
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            if l > m:
                z_mat[l,m] = np.sum(tf[m:(l-1)])
            elif l < m:
                z_mat[l,m] = -np.sum(tf[l:(m-1)])
            else:
                z_mat[l,m] = tf[l]

    Cl = np.ones(len(Cl))
    return dwba_tilted_qx_fast_inner_sk(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)


@numba.jit(nopython=True, cache=False)
def dwba_tilted_qx_fast_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl):
    sum = np.empty(qx.shape, dtype=np.complex128)
    for i in xrange(len(qx)):
        n1 = n[:-1,i]

        size = len(n1)
        o = np.empty((size,size), dtype=np.complex128)
        
        for l in xrange(size):
           for r in xrange(size):
               o[l,r] = ((n[l,i]**2-n[l+1,i]**2)*(t1[l,i]+r1[l,i])*(t2[l,i]+r2[l,i])) * np.conj((n[r,i]**2-n[r+1,i]**2)*(t1[r,i]+r1[r,i])*(t2[r,i]+r2[r,i]))

        sum[i] = np.sum(o*np.exp(-np.abs(z_mat)*qx[i]**2/xi_perp)*Cl[i]*np.exp(-1j*qx[i]*np.tan(np.radians(beta))*z_mat))
            
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum

@numba.jit(nopython=True, cache=False)
def dwba_tilted_qx_fast_inner_sk(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl):
    sum = np.empty(qx.shape, dtype=np.complex128)
    for i in xrange(len(qx)):
        n1 = n[:-1,i]

        size = len(n1)
        o = np.empty((size,size), dtype=np.complex128)
        
        for l in xrange(size):
           for r in xrange(size):
               o[l,r] = ((n[l,i]**2-n[l+1,i]**2)*(t1[l,i])*(t2[l,i])) * np.conj((n[r,i]**2-n[r+1,i]**2)*(t1[r,i])*(t2[r,i]))

        sum[i] = np.sum(o*np.exp(-np.abs(z_mat)*qx[i]**2/xi_perp)*Cl[i]*np.exp(-1j*qx[i]*np.tan(np.radians(beta))*z_mat))
            
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum

def dwba_tilted_qx_fast_me(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
    r1,t1,r2,t2 = res
    r1,t1,r2,t2 =r1[:-1,:],t1[:-1,:],r2[:-1,:],t2[:-1,:]
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    tf = t[:,0]
    
    def zz(l,m):
        return z(l,m,tf)

    z_mat = np.fromfunction(np.vectorize(zz),(len(t1),len(t1)))
    Cl = 1.0
    
    sum_arr = []
    for i in xrange(len(qx)):
        n1 = n[:-1,i]
        n2 = n[1:,i]
        
        left = (n1**2-n2**2)*(t1[:,i]+r1[:,i])*(t2[:,i]+r2[:,i])
        right = np.conj(left)
        
        o = np.outer(left,right)

        sum = np.sum(o*C_perp_qxi_zmat(z_mat,xi_perp,qx[i])*Cl*np.exp(-1j*qx[i]*np.tan(np.radians(beta))*z_mat))
        sum_arr.append(sum)
    sum_arr = np.array(sum_arr)
            
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum_arr

def dwba_tilted_qx_verticalOnly(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
    r1,t1,r2,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    tf = t[:,0]
    t1t2 = t1*t2
    t1r2 = t1*r2
    r1t2 = r1*t2
    r1r2 = r1*r2
    
    t1t2c = np.conj(t1t2)
    t1r2c = np.conj(t1r2)
    r1t2c = np.conj(r1t2)
    r1r2c = np.conj(r1r2)
    
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += ((n[m]**2-n[m+1]**2)*np.conj(n[l]**2-n[l+1]**2)*(
                        (t1t2[l]*(t1t2c[m])+r1r2[l]*(r1r2c[m]))
                       +(t1t2[l]*(r1r2c[m])+r1r2[l]*(t1t2c[m]))
                       +(t1r2[l]*(t1t2c[m])+r1t2[l]*(r1r2c[m]))
                       +(r1t2[l]*(t1t2c[m])+t1r2[l]*(r1r2c[m]))
                       +(t1t2[l]*(t1r2c[m])+r1r2[l]*(r1t2c[m]))
                       +(r1r2[l]*(t1r2c[m])+t1t2[l]*(r1t2c[m]))
                       +(t1r2[l]*(t1r2c[m])+r1t2[l]*(r1t2c[m]))
                       +(r1t2[l]*(t1r2c[m])+t1r2[l]*(r1t2c[m]))
                    )*C_perp_qx(l,m,tf,xi_perp,qx)*1.0*np.exp(-1j*qx*np.tan(np.radians(beta))*z(l,m,tf)))
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum


def dwba_tilted_qx_sk(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
    r1,t1,r2,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    tf = t[:,0]
    t1t2 = t1*t2
    t1r2 = t1*r2
    r1t2 = r1*t2
    r1r2 = r1*r2
    
    t1t2c = np.conj(t1t2)
    t1r2c = np.conj(t1r2)
    r1t2c = np.conj(r1t2)
    r1r2c = np.conj(r1r2)
    
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += ((n[m]**2-n[m+1]**2)*np.conj(n[l]**2-n[l+1]**2)*(
                        (t1t2[l]*(t1t2c[m]))
                    )*C_perp_qx(l,m,tf,xi_perp,qx)*C_lateral(qx, sigma, xi_lat, H)*np.exp(-1j*qx*np.tan(np.radians(beta))*z(l,m,tf)))
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum


def z(l,m,thickness):
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])
    elif l < m:
                sum = -np.sum(thickness[l:(m-1)])
    else:
                sum = thickness[l]
    return sum

def dwba_gisaxs_qx(res, qx, qy, t,n,wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    tf = t[:,0]
    t1t2 = t1*t2
    t1r2 = t1*r2
    r1t2 = r1*t2
    r1r2 = r1*r2
    
    t1t2c = np.conj(t1t2)
    t1r2c = np.conj(t1r2)
    r1t2c = np.conj(r1t2)
    r1r2c = np.conj(r1r2)
    
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += ((n[m]**2-n[m+1]**2)*np.conj(n[l]**2-n[l+1]**2)*(
                        (t1t2[l]*(t1t2c[m])+r1r2[l]*(r1r2c[m]))
                       +(t1t2[l]*(r1r2c[m])+r1r2[l]*(t1t2c[m]))
                       +(t1r2[l]*(t1t2c[m])+r1t2[l]*(r1r2c[m]))
                       +(r1t2[l]*(t1t2c[m])+t1r2[l]*(r1r2c[m]))
                       +(t1t2[l]*(t1r2c[m])+r1r2[l]*(r1t2c[m]))
                       +(r1r2[l]*(t1r2c[m])+t1t2[l]*(r1t2c[m]))
                       +(t1r2[l]*(t1r2c[m])+r1t2[l]*(r1t2c[m]))
                       +(r1t2[l]*(t1r2c[m])+t1r2[l]*(r1t2c[m]))
                    )*C_perp_qx(l,m,tf,xi_perp,np.sqrt(np.abs(qx)**2+np.abs(qy)**2))*C_lateral(np.sqrt(np.abs(qx)**2+np.abs(qy)**2), sigma, xi_lat, H))
    return sum

#def dwba_verticalOnly(res, t,n,wl, angle_in, xi_perp,qx):
    #r1,r2,t1,t2 = res
    ##qz1, qz2, qz3, qz4 = qz
   ## H=hurst
    #sum = 0
    #for l in xrange(len(t1)):
        #for m in xrange(len(t1)):
            #sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        #(t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       #+(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       #+(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       #+(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       #+(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       #+(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       #+(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       #+(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    #)*C_perp_qx(l,m,t[:,0],xi_perp,qx)
    #return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*sum