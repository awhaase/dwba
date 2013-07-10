import numpy as np
from integrals import *
from matrixmethod import *

def dwba(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    xi_lat = 5.5
    xi_perp = 400
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[l,0]**2-n[l+1,0]**2)*(
                        (t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz1[m])
                       +(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz4[m])
                       +(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz2[m])
                       +(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz3[m])
                       +(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz1[m])
                       +(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz4[m])
                       +(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz2[m])
                       +(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz3[m])
                    )*C_perp(l,m,t[:,0],xi_perp)
    return (1/(16*np.pi**2))*sum


def dwba_sk(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    xi_lat = 5.5
    xi_perp =400
    sum = 0
    for l in xrange(len(t1)):
            for m in xrange(len(t1)):
                sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[l,0]**2-n[l+1,0]**2)*(
                            t1[l]*t2[l]*np.conj(t1[m]*t2[m])
                            )*C_lateral(qx, 0.5, xi_lat, qz1[m], qz1[l])*C_perp(l,m,t[:,0],xi_perp)
    #else:
        #for m in xrange(len(t1)):
                #sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[m,0]**2-n[m+1,0]**2)*(
                            #t1[m]*t2[m]*np.conj(t1[m]*t2[m])
                            #)*C_lateral(qx, 0.5, xi_lat, qz1[m], qz1[m])
    return (1/(16*np.pi**2))*sum

def dwba_dynonly(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    #xi_lat = 5.5
    #xi_perp = 400
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[l,0]**2-n[l+1,0]**2)*(
                        (r1[l]*r2[l]*np.conj(r1[m]*r2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz1[m])
                       +(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz4[m])
                       +(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz2[m])
                       +(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))*C_lateral(qx, 0.5, xi_lat, qz1[l], qz3[m])
                       +(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz1[m])
                       +(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz4[m])
                       +(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz2[m])
                       +(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))*C_lateral(qx, 0.5, xi_lat, qz2[l], qz3[m])
                    )*C_perp(l,m,t[:,0],xi_perp)
    return (1/(16*np.pi**2))*sum