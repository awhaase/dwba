import numpy as np
from integrals import *
from matrixmethod import *

#def dwba_old(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    #r1,r2,t1,t2 = res
    #qz1, qz2, qz3, qz4 = qz
    #sum = 0
    #for l in xrange(len(t1)):
        #for m in xrange(len(t1)):
            #sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[l,0]**2-n[l+1,0]**2)*(
                        #(t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       #+(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       #+(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       #+(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       #+(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       #+(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       #+(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       #+(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    #)*C_perp(l,m,t[:,0],xi_perp)*C_lateral(qx, 0.5, xi_lat, )
    #return (1/(16*np.pi**2))*sum

def dwba(res, qx, t,n,wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        (t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    )*C_perp(l,m,t[:,0],xi_perp)*C_lateral(qx, sigma, xi_lat, H)*np.exp(-1j*qx*np.tan(np.radians(0))*z(l,m,t[:,0]))
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*sum

def dwba_tilted(res, qx, t, n, wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma, beta):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        (t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    )*C_perp(l,m,t[:,0],xi_perp)*C_lateral(qx, sigma, xi_lat, H)*np.exp(-1j*qx*np.tan(np.radians(beta))*z(l,m,t[:,0]))
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*sum

def z(l,m,thickness):
    sum = 0
    if l > m:
                sum = np.sum(thickness[m:(l-1)])
    elif l < m:
                sum = -np.sum(thickness[l:(m-1)])
    else:
                sum = thickness[l]
    return sum

def dwba_gisaxs(res, qx, qy, t,n,wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        (t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    )*C_perp(l,m,t[:,0],xi_perp)*C_lateral(np.sqrt(np.abs(qx)**2+np.abs(qy)**2), sigma, xi_lat, H)
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*sum

def dwba_verticalOnly(res, t,n,wl, angle_in, xi_perp):
    r1,r2,t1,t2 = res
    #qz1, qz2, qz3, qz4 = qz
   # H=hurst
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        (t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       +(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       +(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    )*C_perp(l,m,t[:,0],xi_perp)
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*sum

def dwba_sk(res, qx, t,n,wl, qz, xi_lat, xi_perp, angle_in, hurst, sigma):
    r1,r2,t1,t2 = res
    qz1, qz2, qz3, qz4 = qz
    H=hurst
    sum = 0
    for l in xrange(len(t1)):
        for m in xrange(len(t1)):
            sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        (t1[l]*t2[l]*np.conj(t1[m]*t2[m]))
                         )*C_perp(l,m,t[:,0],xi_perp)*C_lateral(qx, sigma, xi_lat, H)
    return (1/np.cos(np.radians(angle_in)))*(1/(16*np.pi**2))*sum

#def dwba_dyn(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    #r1,r2,t1,t2 = res
    #qz1, qz2, qz3, qz4 = qz
    #H=1.0
    #sum = 0
    #for l in xrange(len(t1)):
        #for m in xrange(len(t1)):
            #sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        #(r1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       #+(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))
                       #+(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))
                       #+(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))
                       #+(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                       #+(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       #+(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))
                       #+(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))
                    #)*C_perp(l,m,t[:,0],xi_perp)*C_lateral(qx, 0.5, xi_lat, H)
    #return (1/(16*np.pi**2))*sum


#def dwba_sk(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    #r1,r2,t1,t2 = res
    #qz1, qz2, qz3, qz4 = qz
    #H = 1.0
    #sum = 0
    #for l in xrange(len(t1)):
            #for m in xrange(len(t1)):
                #sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                            #t1[l]*t2[l]*np.conj(t1[m]*t2[m])
                            #)*C_lateral(qx, 0.5, xi_lat,H)*C_perp(l,m,t[:,0],xi_perp)
    ##else:
        ##for m in xrange(len(t1)):
                ##sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[m,0]**2-n[m+1,0]**2)*(
                            ##t1[m]*t2[m]*np.conj(t1[m]*t2[m])
                            ##)*C_lateral(qx, 0.5, xi_lat, qz1[m], qz1[m])
    #return (1/(16*np.pi**2))*sum

#def dwba_qz(res, qx, t,n,wl, qz, xi_lat, xi_perp, hurst):
    #r1,r2,t1,t2 = res
    #qz1, qz2, qz3, qz4 = qz
    #H = hurst
    #sum = 0
    #for l in xrange(len(t1)):
        #for m in xrange(len(t1)):
            #sum += (2*np.pi/wl)**4*(n[m,:]**2-n[m+1,:]**2)*np.conj(n[l,:]**2-n[l+1,:]**2)*(
                        #(t1[l]*t2[l]*np.conj(t1[m]*t2[m])+r1[l]*r2[l]*np.conj(r1[m]*r2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz1[l], qz1[m])
                       #+(t1[l]*t2[l]*np.conj(r1[m]*r2[m])+r1[l]*r2[l]*np.conj(t1[m]*t2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz1[l], qz4[m])
                       #+(t1[l]*r2[l]*np.conj(t1[m]*t2[m])+r1[l]*t2[l]*np.conj(r1[m]*r2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz1[l], qz2[m])
                       #+(r1[l]*t2[l]*np.conj(t1[m]*t2[m])+t1[l]*r2[l]*np.conj(r1[m]*r2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz1[l], qz3[m])
                       #+(t1[l]*t2[l]*np.conj(t1[m]*r2[m])+r1[l]*r2[l]*np.conj(r1[m]*t2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz2[l], qz1[m])
                       #+(r1[l]*r2[l]*np.conj(t1[m]*r2[m])+t1[l]*t2[l]*np.conj(r1[m]*t2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz2[l], qz4[m])
                       #+(t1[l]*r2[l]*np.conj(t1[m]*r2[m])+r1[l]*t2[l]*np.conj(r1[m]*t2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz2[l], qz2[m])
                       #+(r1[l]*t2[l]*np.conj(t1[m]*r2[m])+t1[l]*r2[l]*np.conj(r1[m]*t2[m]))*C_lateral_qz(qx, 0.5, xi_lat, H, qz2[l], qz3[m])
                    #)*C_perp(l,m,t[:,0],xi_perp)
    #return (1/(16*np.pi**2))*sum

#def dwba_qz_sk(res, qx, t,n,wl, qz, xi_lat, xi_perp):
    #r1,r2,t1,t2 = res
    #qz1, qz2, qz3, qz4 = qz
    #H = 1.0
    #sum = 0
    #for l in xrange(len(t1)):
        #for m in xrange(len(t1)):
            #sum += (2*np.pi/wl)**4*(n[m,0]**2-n[m+1,0]**2)*np.conj(n[l,0]**2-n[l+1,0]**2)*(
                        #t1[l]*t2[l]*np.conj(t1[m]*t2[m])*C_lateral_qz(qx, 0.5, xi_lat, H, qz1[l], qz1[m])
                    #)*C_perp(l,m,t[:,0],xi_perp)
    #return (1/(16*np.pi**2))*sum