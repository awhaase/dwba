import numpy as np

def C_lateral(qx, sigma, xi_lat, H):
    return (4*np.pi*H*(sigma*xi_lat)**2/(1+qx**2*xi_lat**2)**(1+H))

def C_perp_qx(l, m, thickness, xi_perp, qx):
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])*qx**2/xi_perp
    elif l < m:
                sum = np.sum(thickness[l:(m-1)])*qx**2/xi_perp
    else:
                sum = thickness[l]*qx**2/xi_perp
    return np.exp(-sum)

def z(l,m,thickness):
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])
    elif l < m:
                sum = -np.sum(thickness[l:(m-1)])
    else:
                sum = thickness[l]
    return sum



def dwba_tilted_qx_cython(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
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
                    )*C_perp_qx(l,m,tf,xi_perp,qx)*C_lateral(qx, sigma, xi_lat, H)*np.exp(-1j*qx*np.tan(np.radians(beta))*z(l,m,tf)))
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*(2*np.pi/wl)**4*sum