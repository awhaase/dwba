import qspace
import numpy as np

def IntegratedProfile(data, angle_in, angle_out, wl, qz, d_qz):
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
        #print "limits: ", d_qzi_l, d_qzi_u
        profile[i] = np.mean(data[i,d_qzi_u:d_qzi_l])
        qxpos[i] = qxd[qzi]
    
    return qxpos, profile