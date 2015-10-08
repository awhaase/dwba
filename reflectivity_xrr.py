import matrixmethod_xrr
import numpy as np
from helper import *

def xrr(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, capthicknesses, cap, capdensities):
    """
    Returns the reflectivity curve of a multilayer.
    
    AOI:                 angle of incidence in degrees from normal
    wavelengths:         array of wavelength at which to calculate the reflectivity
    thicknesses:         array of thicknesses for the individual layers in the multilayer in a single period
    compounds:           array of compund names with the same lengths as thicknesses containing the optical constants for the specified layers
    periods:             number of periods (replications)
    sigma:               mean square roughness
    substrate:           compound name of file with Henke data of substrate
    densities:           array of material densities ( len(densities) == len(compounds) )
    capthicknesses:      additional capping layers deviating in periodicity from thicknesses
    cap:                 array of compund names for files with Henke date for caplayers
    capdensities:        array of densities of caplayers ( len(caphenkes) == len(capdensities) )
    
    
    returns:      reflectivity at the surface
    """
    
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
        n = np.concatenate([[np.array(n_vac)],np.array(LayerCap),n,[henkeSubstrate]])
        t = np.concatenate([np.array(capthicknesses),t,[0]])
    
    
    
    
    t = np.outer(t,np.ones(len(wavelengths)))
    
    #actual calculation
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    r, trans, em, ep = matrixmethod_xrr.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t, sigma)

    return r, trans, kz_AOI