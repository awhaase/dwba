import matrixmethod
import matrixmethod_xrr
reload(matrixmethod_xrr)
import numpy as np
from helper import *
from scipy.special import erf

def generate_mixlayers(n, t, intermix, sigma, mixlayers):
    new_n = [np.array(n[0])]
    new_t = [np.array(t[0])]
    t = np.array(t, dtype=float)
    t = t.reshape(len(t))

    x = np.linspace(-1,1,mixlayers)
    grad_func = np.array((np.sin(x*np.pi/2)+1)*0.5)
    grad_func = grad_func.reshape((len(grad_func),1))

    #vacuum border case
    #n_intermix = (n[0]*(1-intermix[0]) + n[1]*(intermix[0]) )
    #grad_n = (1-grad_func)*new_n[-1] + grad_func*n_intermix
    #grad_t = np.ones(mixlayers)*(sigma[0]/mixlayers)
    #new_n.append(grad_n)
    #new_n.append(np.array(n_intermix))
    #new_t.append(grad_t)
    #if any((t[0]-sigma[0]/2-sigma[1]/2)<0):
                #raise ValueError('Layer thickness can not be smaller than zero')
    #new_t.append(np.array(t[0]-sigma[0]/2-sigma[1]/2))


    #intermediate layers
    for layer in xrange(1,n.shape[0]-1):
        w1a = t[layer-1]/(t[layer-1]+t[layer])
        w1b = t[layer]/(t[layer-1]+t[layer])
        w2a = t[layer]/(t[layer]+t[layer+1])
        w2b = t[layer+1]/(t[layer]+t[layer+1])
        n_intermix = 0.5* (n[layer-1]*intermix[layer-1]*w1a + n[layer]*(1-intermix[layer-1])*w1b + n[layer]*(1-intermix[layer])*w2a + n[layer+1]*(intermix[layer])*w2b )
        grad_n = (1-grad_func)*new_n[-1] + grad_func*n_intermix
        grad_t = np.ones(mixlayers)*(sigma[layer-1]/mixlayers)
        new_n.append(grad_n)
        new_t.append(grad_t)
        new_n.append(np.array([n_intermix]))
        if (t[layer]-sigma[layer-1]/2-sigma[layer]/2)<0:
                raise ValueError('Layer thickness can not be smaller than zero')
        new_t.append(np.array([t[layer]-sigma[layer-1]/2-sigma[layer]/2]))


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

def generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities):
    n_vac = np.ones(len(wavelengths))
    henkeSubstrate = HenkeDataPD(substrate, np.array(wavelengths)).n
    n = []
    t = []
    
    henkeLayer = []
    for i in xrange(len(compounds)):
        layer = compounds[i]
        density = densities[i]
        mat = HenkeDataPD(layer, wavelengths)
        henkeLayer.append(mat.n)
        
    if capthicknesses is not None:
        LayerCap = []
        for i in xrange(len(cap)):
            layer = cap[i]
            density = capdensities[i]
            henke = HenkeDataPD(layer, wavelengths)
            LayerCap.append(henke.n)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
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

def fields(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities):
   
    n,t = generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities)
    
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    refl, trans = matrixmethod_xrr.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t[1:], roughness)
    return refl, trans, n, t

def fields_dwba(AOI, AOE, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix,  mixlayers, capthicknesses, cap,capdensities):
    n,t = generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities)
    
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    kz_AOI, kx_AOI = k_z_generator(np.array(AOE),np.array(wavelengths),n)
    r1, t1 = matrixmethod_xrr.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t[1:], roughness)
    r2, t2 = matrixmethod_xrr.amplitudes(n, np.array(wavelengths), np.array(kz_AOE), t[1:], roughness)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return r1, t1, r2, t2, qx, qz, t, n
    

def xrr(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness,  substrate, densities, intermix, mixlayers, capthicknesses, cap,capdensities):
    re, tr ,n, t = fields(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix, mixlayers, capthicknesses, cap,capdensities)
    return np.abs(re[0])**2
    

    
    
    