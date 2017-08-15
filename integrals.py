import numpy as np
import numba

@numba.jit(nopython=True, cache=False)
def C_lateral(qx, sigma, xi_lat, H):
    """
    Power spectral density function for given parameters.
    :param qx: q_x coordinate at which to calculate the PSD (numpy array)
    :param sigma: rms roughness value (double)
    :param xi_lat: lateral correlation length (double)
    :param H: Hurst factor (double)
    :return: PSD function for given q_x range
    """
    return (4*np.pi*H*(sigma*xi_lat)**2/(1+qx**2*xi_lat**2)**(1+H))

@numba.jit(nopython=True, cache=False)
def C_lateral_qz(qx, sigma, xi_lat, H, qz1, qz2):
    """
    deprecated
    :param qx:
    :param sigma:
    :param xi_lat:
    :param H:
    :param qz1:
    :param qz2:
    :return:
    """
    return (4*np.pi*H*(sigma*xi_lat)**2/(1+qx**2*xi_lat**2)**(1+H))*np.exp(-0.5*(qz1**2+np.conj(qz2)**2)*sigma**2)

@numba.jit(nopython=True, cache=False)
def C_perp(l, m, thickness, xi_perp):
    """
    Roughness replication factor for vertical roughness correlation between two layers l and m independent of
    q_x (deprecated)
    :param l:
    :param m:
    :param thickness:
    :param xi_perp:
    :return:
    """
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])/xi_perp
    elif l < m:
                sum = np.sum(thickness[l:(m-1)])/xi_perp
    else:
                sum = thickness[l]/xi_perp
    return np.exp(-sum)

@numba.jit(nopython=True, cache=False)
def C_perp_qx(l, m, thickness, xi_perp, qx):
    """
    Roughness replication factor for vertical roughness correlation between two layers l and m dependend on
    q_x. Used for dwba calculation.
    :param l: index of layer l
    :param m: index of layer m
    :param thickness: thickness array of the multilayer stack
    :param xi_perp: perpendicular roughness correlation parameter
    :param qx: numpy array of q_x coordinates at which to calculate the correlation function.
    :return: vertical roughness correlation factor for given q_x values and parameters.
    """
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])*qx**2/xi_perp
    elif l < m:
                sum = np.sum(thickness[l:(m-1)])*qx**2/xi_perp
    else:
                sum = thickness[l]*qx**2/xi_perp
    return np.exp(-sum)

@numba.jit(nopython=True, cache=False)
def C_perp_qx_zmat(zmat, xi_perp, qx):
    """
    Numerically improved private function used in fast dwba calculation. Differs from C_perp_qx by accepting a full
    thickness matrix zmat instead of calculating distance between layer l and m every time.
    :param zmat: precalculated matrix with distances between all layers in the multilayer stack
    :param xi_perp: perpendicular roughness correlation parameter
    :param qx: numpy array of q_x coordinates at which to calculate the correlation function.
    :return: vertical roughness correlation factor for given q_x values and parameters.
    """
    return np.exp(-abs(zmat[:,:,np.newaxis])*qx**2/xi_perp)

@numba.jit(nopython=True, cache=False)
def C_perp_qxi_zmat(zmat, xi_perp, qxi):
    """
    deprecated
    :param zmat:
    :param xi_perp:
    :param qxi:
    :return:
    """
    return np.exp(-np.abs(zmat)*qxi**2/xi_perp)