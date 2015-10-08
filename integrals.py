import numpy as np


def C_lateral(qx, sigma, xi_lat, H):
    return (4*np.pi*H*(sigma*xi_lat)**2/(1+qx**2*xi_lat**2)**(1+H))

def C_lateral_qz(qx, sigma, xi_lat, H, qz1, qz2):
    return (4*np.pi*H*(sigma*xi_lat)**2/(1+qx**2*xi_lat**2)**(1+H))*np.exp(-0.5*(qz1**2+np.conj(qz2)**2)*sigma**2)

def C_perp(l, m, thickness, xi_perp):
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])/xi_perp
    elif l < m:
                sum = np.sum(thickness[l:(m-1)])/xi_perp
    else:
                sum = thickness[l]/xi_perp
    return np.exp(-sum)

def C_perp_qx(l, m, thickness, xi_perp, qx):
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])*qx**2/xi_perp
    elif l < m:
                sum = np.sum(thickness[l:(m-1)])*qx**2/xi_perp
    else:
                sum = thickness[l]*qx**2/xi_perp
    return np.exp(-sum)

def C_perp_qx_zmat(zmat, xi_perp, qx):
    return np.exp(-abs(zmat[:,:,np.newaxis])*qx**2/xi_perp)

def C_perp_qxi_zmat(zmat, xi_perp, qxi):
    return np.exp(-abs(zmat)*qxi**2/xi_perp)