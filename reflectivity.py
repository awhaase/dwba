import matrixmethod
import matrixmethod_xrr
import numpy as np
from helper import *

def fields(AOI, AOE, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile):
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
    kz_AOE, kx_AOE = k_z_generator(AOE,np.array(wavelengths),n)
    r1, t1, r2, t2 = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), np.array(kz_AOE), t, sigma)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return (r1, t1, r2, t2, qx, qz, t, n)

def fields_ronly(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile):
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
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    r = matrixmethod_xrr.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t, sigma)
    
    return r

def xrr(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile):
    r = fields_ronly(AOI, wavelengths, thicknesses, henkeFiles, periods, sigma, substrateHenkeFile)
    return np.abs(r)**2
        
    
    
    