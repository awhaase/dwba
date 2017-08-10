import numpy as np
import integrals
import numba

@numba.jit(nopython=True, cache=False)
def dwba(res, wl, angle_in, xi_lat, xi_perp, hurst, sigma, beta):
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
    return dwba_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)

@numba.jit(nopython=True, cache=False)
def dwba_mhe(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
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
    return dwba_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)

@numba.jit(nopython=True, cache=False)
def dwba_mhe_sk(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
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
    return dwba_inner_sk(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)


@numba.jit(nopython=True, cache=False)
def dwba_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl):
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
def dwba_inner_sk(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl):
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
