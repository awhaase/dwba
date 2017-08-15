import numpy as np
from pylab import figure, colorbar
from scipy.interpolate import griddata

def calcQ(Theta_in, Theta_out, wl, refracDelta=None):
        """
        Private function. Calculate the q coordinates for given image array and experimental
        parameters.
        """

        K0 = 2*np.pi*np.ones(np.shape(wl))/wl
            
        af   = np.radians(90.0 - Theta_out)
        ai   = np.radians(90.0 - Theta_in)
        th_f = 0.0
        
        #print af
        
        if refracDelta:
            real_af = np.arcsin(np.sqrt(np.sin(af)**2 - [2*d for d in refracDelta] + [d**2 for d in refracDelta]))
            real_ai = np.arcsin(np.sqrt(np.sin(ai)**2 - [2*d for d in refracDelta] + [d**2 for d in refracDelta]))
        else:
            real_af = af
            real_ai = ai

        #print 'real af:', 90-real_af[::3]*180/np.pi
        #print 'real_ai:', 90-real_ai[::3]*180/np.pi
        
        #determine q coordinates for all pixels
        qx = K0 * ( np.cos(th_f) * np.cos(real_af) - np.cos(real_ai) )
        qy = K0 * ( np.sin(th_f) * np.cos(real_af) )
        qz = K0 * ( np.sin(real_af) + np.sin(real_ai) )
        
        return (qx, qy, qz, real_af, real_ai)
    
def interpolate(qx_in, qz_in, imgArrayPoints):
    """
    Private function. Perform q_x, q_z interpolation for dwba data calculated in angles.
    :param qx_in: q_x array
    :param qz_in: q_z array
    :param imgArrayPoints: measured or calculated diffuse scattering intensities at this point
    :return: interpolated map with qx_coord and qz_coord arrays for corresponding coordinates.
    """
    qx_min=np.min(qx_in)
    qx_max=np.max(qx_in)
    qz_min=np.max(qz_in)
    qz_max=np.min(qz_in)
    q_dim_z = 1600
    q_dim_x = 1600
    qx = qx_in
    qz = qz_in    
    qz_coord, qx_coord = np.mgrid[qz_max:qz_min:complex(0,q_dim_z), qx_min:qx_max:complex(0,q_dim_x)]
    points =[[qx[i],qz[i]] for i in xrange(len(qx))]
    imgArrayQSpace = griddata(points, imgArrayPoints, (qx_coord, qz_coord), method='cubic')
    return imgArrayQSpace, qx_coord, qz_coord


def image(data, angle_in, angle_out, wavelengths):
    """
    Interpolates given diffuse scattering data measured or calculated for specified angles and wavelenghts and returns
    them in q_x and q_z coordinate system.
    :param data: diffuse scattering data
    :param angle_in: corrsponding incidence angles
    :param angle_out: correspoding exit angles
    :param wavelengths: corresponding wavelengths
    :return: map in q-space, grid of q_x values, grid of q_z values
    """
    qx=[]
    qz=[]
    
    for j in xrange(len(angle_in)):
        t_qx, t_qy, t_qz, af, ai = calcQ(angle_in[j], angle_out[j], wavelengths)
        qx.append(t_qx)
        qz.append(t_qz)
    qx = np.array(qx)
    qz = np.array(qz)
    qmap, qxgrid, qzgrid = interpolate(qx.flatten(),qz.flatten(),data.flatten())

    return qmap, qxgrid, qzgrid