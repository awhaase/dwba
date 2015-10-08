import matrixmethod
import matrixmethod_xrr
reload(matrixmethod_xrr)
import numpy as np
from helper import *
from scipy.special import erf

def fields(AOI, AOE, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile, capthicknesses, caphenkes):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:          angle of incidence in degrees from normal
    wavelenths:   array of wavelength at which to calculate the reflectivity
    thicknesses:  array of thicknesses for the individual layers in the multilayer in a single period
    henkeFiles:   array of Henke data files with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:      number of periods
    sigma:        mean square roughness
    
    
    returns:      reflectivity array with dim wavelengths
    """
    
    #preparations
    n_vac = []
    henkeSubstrate = []
    sub = HenkeData(substrateHenkeFile)
    for l in wavelengths:
        n_vac.append(1)
        henkeSubstrate.append((1-sub.getDelta(l))  +sub.getBeta(l)*1j)
    n = []
    t = []
    
    henkeLayer = []
    for layerHenkeFile in henkeFiles:
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)) + henke.getBeta(l)*1j)
        henkeLayer.append(henkeWavelength)
        
    henkeLayerCap = []
    for layerHenkeFile in caphenkes:
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)) + henke.getBeta(l)*1j)
        henkeLayerCap.append(henkeWavelength)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
        periods -= 1
    n = np.array(n)
    n = np.concatenate(n)
    n = np.concatenate([[np.array(n_vac)],[np.array(henkeLayerCap)],n,[np.array(henkeSubstrate)]])
    
    
    t = (np.array(t)).flatten()
    t = np.concatenate([capthicknesses,t,[0]])
    t = np.outer(t,np.ones(len(wavelengths)))
    
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(AOI,np.array(wavelengths),n)
    kz_AOE, kx_AOE = k_z_generator(AOE,np.array(wavelengths),n)
    r1, r2, t1, t2 = matrixmethod.amplitudes_debeyewaller(n, np.array(wavelengths), np.array(kz_AOI), np.array(kz_AOE), t, sigma)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return (r1, t1, r2, t2, qx, qz, t, n)

def fields_debeyewaller(AOI, AOE, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile, capthicknesses, caphenkes):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:          angle of incidence in degrees from normal
    wavelenths:   array of wavelength at which to calculate the reflectivity
    thicknesses:  array of thicknesses for the individual layers in the multilayer in a single period
    henkeFiles:   array of Henke data files with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:      number of periods
    sigma:        mean square roughness
    
    
    returns:      reflectivity array with dim wavelengths
    """
    
    #preparations
    n_vac = []
    henkeSubstrate = []
    sub = HenkeData(substrateHenkeFile)
    for l in wavelengths:
        n_vac.append(1)
        henkeSubstrate.append((1-sub.getDelta(l))  +sub.getBeta(l)*1j)
    n = []
    t = []
    
    henkeLayer = []
    for layerHenkeFile in henkeFiles:
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)) + henke.getBeta(l)*1j)
        henkeLayer.append(henkeWavelength)
        
    henkeLayerCap = []
    for layerHenkeFile in caphenkes:
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)) + henke.getBeta(l)*1j)
        henkeLayerCap.append(henkeWavelength)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
        periods -= 1
    n = np.array(n)
    n = np.concatenate(n)
    n = np.concatenate([[np.array(n_vac)],n,[np.array(henkeSubstrate)]])
    
    
    t = (np.array(t)).flatten()
    t = np.concatenate([t,[0]])
    t = np.outer(t,np.ones(len(wavelengths)))
    
   
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(AOI,np.array(wavelengths),n)
    kz_AOE, kx_AOE = k_z_generator(AOE,np.array(wavelengths),n)
    r1, r2, t1, t2 = matrixmethod.amplitudes_debeyewaller(n, np.array(wavelengths), np.array(kz_AOI), np.array(kz_AOE), t, sigma)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return (r1, t1, r2, t2, qx, qz, t, n)

def fields_debeyewaller_densities(AOI, AOE, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile, densities, capthicknesses, caphenkes, capdensities):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:          angle of incidence in degrees from normal
    wavelenths:   array of wavelength at which to calculate the reflectivity
    thicknesses:  array of thicknesses for the individual layers in the multilayer in a single period
    henkeFiles:   array of Henke data files with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:      number of periods
    sigma:        mean square roughness
    
    
    returns:      reflectivity array with dim wavelengths
    """
        #preparations
    n_vac = []
    henkeSubstrate = []
    sub = HenkeData(substrateHenkeFile)
    for l in wavelengths:
        n_vac.append(1)
        henkeSubstrate.append((1-sub.getDelta(l)) + sub.getBeta(l)*1j)
    n = []
    t = []
    
    henkeLayer = []
    for i in xrange(len(henkeFiles)):
        layerHenkeFile = henkeFiles[i]
        density = densities[i]
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)*density) + henke.getBeta(l)*density*1j)
        henkeLayer.append(henkeWavelength)
        
    if capthicknesses is not None:
        henkeLayerCap = []
        for i in xrange(len(caphenkes)):
            layerHenkeFile = caphenkes[i]
            density = capdensities[i]
            henkeWavelength = []
            henke = HenkeData(layerHenkeFile)
            for l in wavelengths:
                henkeWavelength.append((1-henke.getDelta(l)*density) + henke.getBeta(l)*density*1j)
            henkeLayerCap.append(henkeWavelength)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
        periods -= 1
        
    n = np.array(n)
    n = np.concatenate(n)
    t = (np.array(t)).flatten()
    
    if capthicknesses is None:
        n = np.concatenate([[np.array(n_vac)],n,[np.array(henkeSubstrate)]])
        t = np.concatenate([t,[0]])
    else:
        n = np.concatenate([[np.array(n_vac)],np.array(henkeLayerCap),n,[np.array(henkeSubstrate)]])
        t = np.concatenate([np.array(capthicknesses),t,[0]])
    
    
    
    
    t = np.outer(t,np.ones(len(wavelengths)))
   
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(AOI,np.array(wavelengths),n)
    kz_AOE, kx_AOE = k_z_generator(AOE,np.array(wavelengths),n)
    r1, r2, t1, t2 = matrixmethod.amplitudes_debeyewaller(n, np.array(wavelengths), np.array(kz_AOI), np.array(kz_AOE), t, sigma)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return (r1, t1, r2, t2, qx, qz, t, n)

def fields_debeyewaller_mixlayer(AOI, AOE, wavelengths, thicknesses, henkeFiles, periods, sigma, roughness, substrateHenkeFile, densities, intermix,  mixlayers, capthicknesses, caphenkes,capdensities):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:          angle of incidence in degrees from normal
    wavelenths:   array of wavelength at which to calculate the reflectivity
    thicknesses:  array of thicknesses for the individual layers in the multilayer in a single period
    henkeFiles:   array of Henke data files with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:      number of periods
    sigma:        mean square roughness
    
    
    returns:      reflectivity array with dim wavelengths
    """
    #preparations
    n_vac = []
    henkeSubstrate = []
    sub = HenkeData(substrateHenkeFile)
    for l in wavelengths:
        n_vac.append(1)
        henkeSubstrate.append((1-sub.getDelta(l)) + sub.getBeta(l)*1j)
    n = []
    t = []
    
    henkeLayer = []
    for i in xrange(len(henkeFiles)):
        layerHenkeFile = henkeFiles[i]
        density = densities[i]
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)*density) + henke.getBeta(l)*density*1j)
        henkeLayer.append(henkeWavelength)
        
    if capthicknesses is not None:
        henkeLayerCap = []
        for i in xrange(len(caphenkes)):
            layerHenkeFile = caphenkes[i]
            density = capdensities[i]
            henkeWavelength = []
            henke = HenkeData(layerHenkeFile)
            for l in wavelengths:
                henkeWavelength.append((1-henke.getDelta(l)*density) + henke.getBeta(l)*density*1j)
            henkeLayerCap.append(henkeWavelength)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
        periods -= 1
        
    n = np.array(n)
    n = np.concatenate(n)
    t = (np.array(t)).flatten()
    
    if capthicknesses is None:
        n = np.concatenate([[np.array(n_vac)],n,[np.array(henkeSubstrate)]])
        t = np.concatenate([t,[0]])
    else:
        n = np.concatenate([[np.array(n_vac)],np.array(henkeLayerCap),n,[np.array(henkeSubstrate)]])
        t = np.concatenate([np.array(capthicknesses),t,[0]])
    
    
    
    
    t = np.outer(t,np.ones(len(wavelengths)))  
    
    #introduction of mixlayers
    new_n = [np.array([n[0]])]
    new_t = []
    for layer in xrange(1,n.shape[0]):
        x = np.linspace(-1,1,mixlayers)
        grad_func = np.array((np.sin(x*np.pi/2)+1)*0.5)
        grad_func = grad_func.reshape((len(grad_func),1))
        if layer<(n.shape[0]-1):
            n_intermix = (n[layer,:]*(1-0.5*intermix[layer]) + n[layer+1,:]*0.5*intermix[layer])*0.5+(n[layer,:]*(1-0.5*intermix[layer-1]) + n[layer-1,:]*0.5*intermix[layer-1])*0.5
        else:
            n_intermix = n[layer,:]
        grad_n = (1-grad_func)*new_n[-1] + grad_func*n_intermix
        grad_t = np.ones(mixlayers).reshape(mixlayers,1)*(np.ones(len(t[0]))*sigma[layer-1]/mixlayers)
        new_n.append(grad_n)
        new_t.append(grad_t)
        new_n.append(np.array([n_intermix]))
        if layer<len(sigma):
            new_t.append([t[layer-1,:]-sigma[layer-1]/2-sigma[layer]/2])
            if any((t[layer-1,:]-sigma[layer-1]/2-sigma[layer]/2)<0):
                raise ValueError('Layer thickness can not be smaller than zero')
        else:
            new_t.append([t[layer-1,:]])
    
    new_n = np.concatenate(new_n)
    new_t = np.concatenate(new_t)
    
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(AOI,np.array(wavelengths),new_n)
    kz_AOE, kx_AOE = k_z_generator(AOE,np.array(wavelengths),new_n)
    r1, r2, t1, t2 = matrixmethod.amplitudes_debeyewaller(new_n, np.array(wavelengths), np.array(kz_AOI), np.array(kz_AOE), new_t, roughness)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return (r1, t1, r2, t2, qx, qz, new_t, new_n)

def fields_debeyewaller_gisaxs(AOI, AOE_x, AOE_y, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:          angle of incidence in degrees from normal
    wavelenths:   array of wavelength at which to calculate the reflectivity
    thicknesses:  array of thicknesses for the individual layers in the multilayer in a single period
    henkeFiles:   array of Henke data files with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:      number of periods
    sigma:        mean square roughness
    
    
    returns:      reflectivity array with dim wavelengths
    """
    
    #preparations
    n_vac = []
    henkeSubstrate = []
    sub = HenkeData(substrateHenkeFile)
    for l in wavelengths:
        n_vac.append(1)
        henkeSubstrate.append((1-sub.getDelta(l))  +sub.getBeta(l)*1j)
    n = []
    t = []
    
    henkeLayer = []
    for layerHenkeFile in henkeFiles:
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)) + henke.getBeta(l)*1j)
        henkeLayer.append(henkeWavelength)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
        periods -= 1
    n = np.array(n)
    n = np.concatenate(n)
    n = np.concatenate([[np.array(n_vac)],n,[np.array(henkeSubstrate)]])
    
    
    t = (np.array(t)).flatten()
    t = np.concatenate([t,[0]])
    t = np.outer(t,np.ones(len(wavelengths)))
    
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(AOI,np.array(wavelengths),n)
    kz_AOE, kx_AOE = k_z_generator(AOE_x,np.array(wavelengths),n)
    r1, r2, t1, t2 = matrixmethod.amplitudes_debeyewaller(n, np.array(wavelengths), np.array(kz_AOI), np.array(kz_AOE), t, sigma)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - np.cos(np.radians(AOE_y))*kx_AOE
    qy = np.sin(np.radians(AOE_y))*kx_AOE
    
    return (r1, t1, r2, t2, qx, qy, qz, t, n)

def fields_ronly(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile, densities, capthicknesses, caphenkes, capdensities, full=False):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:          angle of incidence in degrees from normal
    wavelenths:   array of wavelength at which to calculate the reflectivity
    thicknesses:  array of thicknesses for the individual layers in the multilayer in a single period
    henkeFiles:   array of Henke data files with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:      number of periods
    sigma:        mean square roughness
    
    
    returns:      reflectivity array with dim wavelengths
    """
    
    #preparations
    n_vac = []
    henkeSubstrate = []
    sub = HenkeData(substrateHenkeFile)
    for l in wavelengths:
        n_vac.append(1)
        henkeSubstrate.append((1-sub.getDelta(l)) + sub.getBeta(l)*1j)
    n = []
    t = []
    
    henkeLayer = []
    for i in xrange(len(henkeFiles)):
        layerHenkeFile = henkeFiles[i]
        density = densities[i]
        henkeWavelength = []
        henke = HenkeData(layerHenkeFile)
        for l in wavelengths:
            henkeWavelength.append((1-henke.getDelta(l)*density) + henke.getBeta(l)*density*1j)
        henkeLayer.append(henkeWavelength)
        
    if capthicknesses is not None:
        henkeLayerCap = []
        for i in xrange(len(caphenkes)):
            layerHenkeFile = caphenkes[i]
            density = capdensities[i]
            henkeWavelength = []
            henke = HenkeData(layerHenkeFile)
            for l in wavelengths:
                henkeWavelength.append((1-henke.getDelta(l)*density) + henke.getBeta(l)*density*1j)
            henkeLayerCap.append(henkeWavelength)
        
    
    while periods>0:
        n.append(np.array(henkeLayer))
        t.append(np.array(thicknesses))
        periods -= 1
        
    n = np.array(n)
    n = np.concatenate(n)
    t = (np.array(t)).flatten()
    
    if capthicknesses is None:
        n = np.concatenate([[np.array(n_vac)],n,[np.array(henkeSubstrate)]])
        t = np.concatenate([t,[0]])
    else:
        n = np.concatenate([[np.array(n_vac)],np.array(henkeLayerCap),n,[np.array(henkeSubstrate)]])
        t = np.concatenate([np.array(capthicknesses),t,[0]])
    
    
    
    
    t = np.outer(t,np.ones(len(wavelengths)))
    
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    r, trans = matrixmethod_xrr.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t, sigma)
    
    if full:
        return r,n,t
    else:
        return r



def xrr(AOI, wavelengths, thicknesses, henkeFiles, periods,sigma, substrateHenkeFile, densities, capthicknesses, caphenkes, capdensities,full_matrix=False):
    if full_matrix:
        r, n, t = fields_ronly(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile, densities, capthicknesses, caphenkes, capdensities, full_matrix)
    else:
         r = fields_ronly(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile, densities, capthicknesses, caphenkes, capdensities, full_matrix)
    if not full_matrix:
        return np.abs(r[0])**2
    else:
        return np.abs(r)**2, n, t


### Mischschichtenansatz

def fields_ronly_mixlayer(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities):
   
    #preparations
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
        t = np.concatenate([t,[0]])
    else:
        n = np.concatenate([[np.array(n_vac)],np.array(LayerCap),n,[np.array(henkeSubstrate)]])
        t = np.concatenate([np.array(capthicknesses),t,[0]])

    t = np.outer(t,np.ones(len(wavelengths)))
   
     
    #introduction of mixlayers
    new_n = [np.array([n[0]])]
    new_t = []
    for layer in xrange(1,n.shape[0]):
        x = np.linspace(-1,1,mixlayers)
        grad_func = np.array((np.sin(x*np.pi/2)+1)*0.5)
        grad_func = grad_func.reshape((len(grad_func),1))
        
        if layer<(n.shape[0]-1):
            n_intermix = (n[layer,:]*(1-0.5*intermix[layer]) + n[layer+1,:]*0.5*intermix[layer])*0.5+(n[layer,:]*(1-0.5*intermix[layer-1]) + n[layer-1,:]*0.5*intermix[layer-1])*0.5
        else:
            n_intermix = n[layer,:]
        grad_n = (1-grad_func)*new_n[-1] + grad_func*n_intermix
        grad_t = np.ones(mixlayers).reshape(mixlayers,1)*(np.ones(len(t[0]))*sigma[layer-1]/mixlayers)
        new_n.append(grad_n)
        new_t.append(grad_t)
        new_n.append(np.ones(mixlayers).reshape(mixlayers,1)*np.array([n_intermix]))
        if layer<len(sigma):
            new_t.append(np.ones(mixlayers).reshape(mixlayers,1)*(np.array([t[layer-1,:]-sigma[layer-1]/2-sigma[layer]/2])/mixlayers))
            if any((t[layer-1,:]-sigma[layer-1]/2-sigma[layer]/2)<0):
                raise ValueError('Layer thickness can not be smaller than zero')
        else:
            new_t.append(np.ones(mixlayers).reshape(mixlayers,1)*np.array([t[layer-1,:]]))
    
    new_n = np.concatenate(new_n)
    new_t = np.concatenate(new_t)
    
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),new_n)
    refl, trans = matrixmethod_xrr.amplitudes(new_n, np.array(wavelengths), np.array(kz_AOI), new_t, roughness)
    return refl, trans, new_n, new_t



def xrr_mixlayer(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, roughness,  substrateHenkeFile, densities, intermix, mixlayers, capthicknesses, caphenkes,capdensities, full_matrix=False, raw=False):
    re, tr ,n, t = fields_ronly_mixlayer(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, roughness, substrateHenkeFile, densities, intermix, mixlayers, capthicknesses, caphenkes,capdensities)
    if full_matrix:
        return np.abs(re)**2,np.abs(tr)**2 , n, t
    if raw:
        return re,tr , n, t
    else:
        return np.abs(re[0])**2, n, t
    

    
    
    