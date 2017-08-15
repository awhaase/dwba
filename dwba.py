import numpy as np
import integrals
import numba

@numba.jit(nopython=True, cache=False)
def dwba(res, wl, angle_in, xi_lat, xi_perp, hurst, sigma, beta):
    """
    Core function to calculate the fully dynamic diffuse scattering across given wavelengths at specified incidence and
    exit angles. The function returns a single cut for a scalar incidence and exit angle.
    :param res: return value of the reflectivity.fields_dwba function. Contains the specular beam calculations
    :param wl: numpy array with wavelengths in nm at which the diffuse scattering should be evaluated
    :param angle_in: double incidence ange. Note that this angle should correspond to the angle of incidence handed to
                     the reflectivity.fields_dwba function to produce correct results
    :param xi_lat: double with the lateral correlation length for the PSD calculation
    :param xi_perp: double with the vertical correlation length parameter
    :param hurst: double with the Hurst factor for the PSD calculation
    :param sigma: double with the rms roughness value for all interfaces
    :param beta: double angle in degrees for off-normal roughness correlation (beta = 0.0 corresponds to correlation along
                 the surface normal
    :return: numpy array of the differential scattering cross section for the given parameters
    """
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
def dwba_mhe(res, wl, angle_in, xi_lat, xi_perp, hurst, sigma, beta):
    """
    Core function to calculate the fully dynamic multilayer enhancement factor across given wavelengths at specified
    incidence and exit angles. The function returns a single cut for a scalar incidence and exit angle. The PSD parameters
    are incucluded in the function signature for compatibility reasons. They have no effect on the result. The only
    parameters entering the calculation apart from the res, wl and angle_in are xi_perp and beta.
    :param res: return value of the reflectivity.fields_dwba function. Contains the specular beam calculations
    :param wl: numpy array with wavelengths in nm at which the diffuse scattering should be evaluated
    :param angle_in: double incidence ange. Note that this angle should correspond to the angle of incidence handed to
                     the reflectivity.fields_dwba function to produce correct results
    :param xi_lat: double with the lateral correlation length for the PSD calculation
    :param xi_perp: double with the vertical correlation length parameter
    :param hurst: double with the Hurst factor for the PSD calculation
    :param sigma: double with the rms roughness value for all interfaces
    :param beta: double angle in degrees for off-normal roughness correlation (beta = 0.0 corresponds to correlation along
                 the surface normal
    :return: numpy array of the differential scattering cross section for the given parameters
    """
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

    Cl = np.ones(len(Cl))
    return dwba_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)

@numba.jit(nopython=True, cache=False)
def dwba_mhe_sk(res, wl, angle_in, xi_lat, xi_perp, hurst, sigma, beta):
    """
    Core function to calculate the semi-kinematic multilayer enhancement factor across given wavelengths at specified
    incidence and exit angles. The function returns a single cut for a scalar incidence and exit angle. The PSD parameters
    are incucluded in the function signature for compatibility reasons. They have no effect on the result. The only
    parameters entering the calculation apart from the res, wl and angle_in are xi_perp and beta.
    :param res: return value of the reflectivity.fields_dwba function. Contains the specular beam calculations
    :param wl: numpy array with wavelengths in nm at which the diffuse scattering should be evaluated
    :param angle_in: double incidence ange. Note that this angle should correspond to the angle of incidence handed to
                     the reflectivity.fields_dwba function to produce correct results
    :param xi_lat: double with the lateral correlation length for the PSD calculation
    :param xi_perp: double with the vertical correlation length parameter
    :param hurst: double with the Hurst factor for the PSD calculation
    :param sigma: double with the rms roughness value for all interfaces
    :param beta: double angle in degrees for off-normal roughness correlation (beta = 0.0 corresponds to correlation along
                 the surface normal
    :return: numpy array of the differential scattering cross section for the given parameters
    """
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

    Cl = np.ones(len(Cl))
    return dwba_inner_sk(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl)


@numba.jit(nopython=True, cache=False)
def dwba_inner(qx, n, t1, r1, t2, r2, z_mat, xi_perp, Cl, beta, angle_in, wl):
    """
    Private function performing the fully dynbamic dwba calculation in an numerically optimized manner. Is called by
    dwba and dwba_mhe.
    """
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
    """
    Private function performing the semi-kinematic calculation in a numerically optimized manner. Is called by dwba_mhe_sk.
    """
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
