import qspace
import numpy as np

def IntegratedProfile(data, angle_in, angle_out, wl, qz, d_qz, d_a):
    """
    Profile on original data given any qz value.
    """
    profile = np.zeros(np.shape(angle_in)[0], dtype=float)
    qxpos = np.zeros(np.shape(angle_in)[0], dtype=float)
    
    for i in xrange(0,np.shape(angle_in)[0]):
        qxd, qyd, qzd, real_ai, real_af = qspace.calcQ(angle_in[i], angle_out[i], wl)
        #print qzd
        qzi = np.nanargmin(np.abs(qz - np.array(qzd)))
        d_qzi_u = np.nanargmin(np.abs((qz+d_qz) - np.array(qzd)))
        d_qzi_l = np.nanargmin(np.abs((qz-d_qz) - np.array(qzd)))
        #print qzd[qzi]
        qx, blay, blaz, blaai, blaaf  = qspace.calcQ(angle_in[i], angle_out[i], wl[qzi])
        #print qxd[qzi]
        #print "limits: ", d_qzi_u, d_qzi_l
        profile[i] = np.mean(data[i-d_a:i+d_a,d_qzi_u:d_qzi_l])
        qxpos[i] = qxd[qzi]
    
    #Raumwinkel Korrektur
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    
    
    return qxpos, omega*profile

def profileInQ(interpolation, qxgrid, qzgrid, direction, qx, qz, avrgQ, qx_error=0):
        
        #global interpolation, qx_max, qx_min, qz_max, qz_min
        
        position = 'qz'
        avrgPx = 0
        qx_grid, qz_grid = _gridpointForQ(qx,qz, qxgrid, qzgrid)
        
        profile = None

        if direction == 'qx':
            position = qz_grid
            qx_grid, posavrgPx  = _gridpointForQ(qx, qz+avrgQ, qxgrid, qzgrid)
            avrgPx = np.abs(posavrgPx - position)
            print (qz_grid-avrgPx)
            print (qz_grid+avrgPx)
            profile = np.mean(interpolation[(qz_grid-avrgPx):(qz_grid+avrgPx),:],axis=0)
        else:
            position = qx_grid
            posavrgPx, qz_grid  = _gridpointForQ(qx+avrgQ, qz, qxgrid, qzgrid)
            avrgPx = np.abs(posavrgPx - position)
            profile = np.mean(interpolation[:,(qx_grid-avrgPx):(qx_grid+avrgPx)],axis=1)   
            
        #Raumwinkel Korrektur
        r=250.0
        wx=4.5
        wy=4.5
        
        omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
        
        if qx_error is 0:
            return omega*profile
        else:
           pos_qx, qz_grid  = _gridpointForQ(qx, qz, qxgrid, qzgrid)
           pos_qxerror, qz_grid  = _gridpointForQ(qx+qx_error, qz, qxgrid, qzgrid)
           window = np.ones(np.abs(pos_qx-pos_qxerror))/np.abs(pos_qx-pos_qxerror)
           print window.shape
           return omega*np.convolve(profile, window, "same")
    
def _gridpointForQ(qx, qz, qxgrid, qzgrid):
        qx_grid = np.nanargmin(np.abs(qx - qxgrid[0,:]))
        qz_grid = np.nanargmin(np.abs(qz - qzgrid[:,0]))
        
        return (qx_grid, qz_grid)