import numpy as np
from pylab import figure, colorbar
from scipy.interpolate import griddata

def calcQ(Theta_in, Theta_out, wl, refracDelta=None):
        """
        Calculate the q coordinates for given image array and experimental
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


def image(data, angle_in, angle_out, wl, cmap, show=True, aspect=0):
    qx=[]
    qz=[]
    
    #Raumwinkel Korrektur
    r=250.0
    wx=4.0
    wy=4.0
    
    #omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    omega=1
    
    for j in xrange(len(angle_in)):
        t_qx, t_qy, t_qz, af, ai = calcQ(angle_in[j], angle_out[j],wl)
        qx.append(t_qx)
        qz.append(t_qz)
    qx = np.array(qx)
    qz = np.array(qz)
    qmap, qxgrid, qzgrid = interpolate(qx.flatten(),qz.flatten(),data.flatten())
    if show:
        fig = figure()
        ax = fig.add_subplot(111)
        qx_min=np.min(qx.flatten())
        qx_max=np.max(qx.flatten())
        qz_min=np.max(qz.flatten())
        qz_max=np.min(qz.flatten())
        img = ax.imshow(omega*qmap, extent=(qx_min,qx_max,qz_min,qz_max), cmap="gray", aspect=aspect)
        cb = colorbar(img, format="%.0e")
        cb.set_label("Intensity $I/I_0$")
        img.set_cmap(cmap)
        img.get_cmap().set_bad('black')
        #ax.axhspan(0.875,0.89, color='black')
        #ax.axvspan(-0.022,0.022,color='black')
        ax.axis([-0.3,0.3,3.8,4.3])
        ax.set_xlabel('$q_x$ [1/nm]')
        ax.set_ylabel('$q_z$ [1/nm]')
        fig.canvas.draw()
    else:
        fig = None
    #corr = np.multiply(np.array([50,5]),np.array(index, float)) + [0,0.5]
    #text = '$i=%i, \\, \\xi_p = %.1f$ nm, $\\xi_l = %.1f$ nm' % (i,corr[i,0], corr[i,1])
    #x.annotate(text, xy=(-0.28,0.98), color='red')
    return fig, qmap, qxgrid, qzgrid

def image2T(data, angles, wl):
    qx=[]
    qz=[]
    for j in xrange(len(angles)):
        t_qx, t_qy, t_qz, af, ai = calcQ(6.75, angles[j],wl)
        qx.append(t_qx)
        qz.append(t_qz)
    qx = np.array(qx)
    qz = np.array(qz)
    qmap = interpolate(qx.flatten(),qz.flatten(),data.flatten())
    fig = figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(qmap, extent=(-0.3,0.3,0.85,1.0), cmap="gray")
    img.set_cmap("gray")
    img.get_cmap().set_bad('black')
    img.set_clim([0,5E-5])
    ax.axhspan(0.85,0.89, color='black')
    ax.axvspan(-0.022,0.022,color='black')
    ax.axis([-0.3,0.3,0.875,1.0])
    ax.set_xlabel('$q_x$ [1/nm]')
    ax.set_ylabel('$q_z$ [1/nm]')
    fig.canvas.draw()
    #corr = np.multiply(np.array([50,5]),np.array(index, float)) + [0,0.5]
    #text = '$i=%i, \\, \\xi_p = %.1f$ nm, $\\xi_l = %.1f$ nm' % (i,corr[i,0], corr[i,1])
    #x.annotate(text, xy=(-0.28,0.98), color='red')
    return fig, qmap