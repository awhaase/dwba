import matrixmethod
reload(matrixmethod)
import numpy as np
from helper import *
import numba
from scipy.special import erf


def generate_mixlayers(n, t, intermix, sigma, mixlayers):
    new_n = [np.array(n[0])]
    new_t = [np.array(t[0])]
    t = np.array(t, dtype=float)
    t = t.reshape(len(t))

    x = np.linspace(-1,1,mixlayers)
    grad_func = np.array((np.sin(x*np.pi/2)+1)*0.5)
    grad_func = grad_func.reshape((len(grad_func),1))

    #intermediate layers
    for layer in xrange(1,n.shape[0]-1):
        w1a = t[layer-1]/(t[layer-1]+t[layer])
        w1b = t[layer]/(t[layer-1]+t[layer])
        w2a = t[layer]/(t[layer]+t[layer+1])
        w2b = t[layer+1]/(t[layer]+t[layer+1])
        
        #intermixing function defined here
        n_intermix = 0.5* ((w1a*n[layer-1]+w1b*n[layer])*intermix[layer-1] + n[layer]*(1-intermix[layer-1]) +
                           n[layer]*(1-intermix[layer]) + (n[layer]*w2a+n[layer+1]*w2b)*(intermix[layer]) )
        grad_n = (1-grad_func)*new_n[-1] + grad_func*n_intermix
        grad_t = np.ones(mixlayers)*(sigma[layer-1]/mixlayers)
        new_n.append(grad_n)
        new_t.append(grad_t)
        new_n.append(np.array([n_intermix]))
        if (t[layer]-sigma[layer-1]/2-sigma[layer]/2)<0:
                raise ValueError('Layer thickness can not be smaller than zero. Reduce interface widths!')
        new_t.append(np.array(t[layer]-sigma[layer-1]/2-sigma[layer]/2))


    #substrate border case
    n_intermix = n[-1]
    grad_n = (1-grad_func)*new_n[-1] + grad_func*n_intermix
    grad_t = np.ones(mixlayers)*(sigma[-1]/mixlayers)
    new_n.append(grad_n)
    new_t.append(grad_t)
    new_n.append(np.array([n_intermix]))
    new_t.append(np.array([t[-1]]))


    new_n = np.vstack(new_n)
    new_t = np.concatenate(new_t)
    new_t = new_t.reshape(len(new_t),1)
    return new_n, new_t

def generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities, drift=0):
    n_vac = np.ones(len(wavelengths))
    henkeSubstrate = HenkeDataPD(substrate, np.array(wavelengths)).n
    n = []
    t = []
    
    henkeLayer = []
    for i in xrange(len(compounds)):
        layer = compounds[i]
        density = densities[i]
        mat = HenkeDataPD(layer, wavelengths)
        if type(density) == list:
            henkeLayer.append(1-density[0]*mat.getDelta()+1j*density[1]*mat.getBeta())
        else:
            henkeLayer.append(1-density*mat.getDelta()+1j*density*mat.getBeta())
        
    if capthicknesses is not None:
        LayerCap = []
        for i in xrange(len(cap)):
            layer = cap[i]
            density = capdensities[i]
            henke = HenkeDataPD(layer, wavelengths)
            if type(density) == list:
                LayerCap.append(1-density[0]*henke.getDelta()+1j*density[1]*henke.getBeta())
            else:
                LayerCap.append(1-density*henke.getDelta()+1j*density*henke.getBeta())
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses)+(periods*drift-200*drift))
        periods -= 1
        
    n = np.array(n)
    n = np.concatenate(n)
    t = (np.array(t)).flatten()
    
    if capthicknesses is None:
        n = np.concatenate([[np.array(n_vac)],n,[np.array(henkeSubstrate)]])
        t = np.concatenate([[0],t,[0]])
    else:
        n = np.concatenate([[np.array(n_vac)],np.array(LayerCap),n,[np.array(henkeSubstrate)]])
        t = np.concatenate([[0],np.array(capthicknesses),t,[0]])

    #t = np.outer(t,np.ones(len(wavelengths)))
    t = t.reshape(len(t),1)
   
    if mixlayers > 0:
        n,t = generate_mixlayers(n, t, intermix, sigma, mixlayers)
        
    return n,t


def fields(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities, drift=0):
   
    n,t = generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities, drift)
    
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    refl, trans = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t[1:], roughness)
    return refl, trans, n, t

def fields_dwba(AOI, AOE, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix,  mixlayers, capthicknesses, cap,capdensities):
    n,t = generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities)
    
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    kz_AOE, kx_AOE = k_z_generator(np.array(AOE),np.array(wavelengths),n)
    r1, t1 = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t[1:], roughness)
    r2, t2 = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOE), t[1:], roughness)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return r1, t1, r2, t2, qx, qz, t, n
    

def xrr(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness,  substrate, densities, intermix, mixlayers, capthicknesses, cap,capdensities, drift=0):
    re, tr ,n, t = fields(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix, mixlayers, capthicknesses, cap,capdensities, drift)
    return np.abs(re[0])**2
    

    
    
    