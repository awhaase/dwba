import matrixmethod
reload(matrixmethod)
import numpy as np
from helper import *
import numba
from scipy.special import erf


def generate_mixlayers(n, t, intermix, sigma, mixlayers):
    """
    Private function. This method converts a binary or discrete layer stack into a stack with graded interfaces and
    returns the new layer system. For that purpose it requires the optical constants n and thicknesses t, as well as the
    widths of the graded interfaces in sigma and the number of points with which the interface region should be sampled
    given in mixlayers. The interface shape is hard coded to be sinodoidal. Intermix gives the amount of layer
    intermixing through a one dimensional array of values between 0.0 and 1.0 (full intermixing). The length of the
    arrays sigma and intermixing has to correspond to the number of interfaces in the system, i.e. number of layers plus
    one.
    :param n: two dimensional numpy array with indices of refraction
    :param t: one dimensional numpy array with the thicknesses of the individual layers
    :param intermix: one dimensional numpy array with the intermixing fraction of the previous layer with the next.
                     The length has to correspond to the number of interfaces.
    :param sigma: one dimensional numpy array with the interface region thickness for the top interface of the respective
                  layer. The length has to correspond to the number of interfaces.
    :param mixlayers: integer number specifying the number of sampling points for the interfaces.
    :return: tuple (n,t) for the new graded system
    """
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
        
        #intermixing function defined here (how to calculate the mixing of previous and next layer with the current one)
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

def generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities):
    """
    Generates the arrays n and t specifying the layer system for all other functions in this project. It requires  the
    user to give the multilayer system using the given experimental and model parameters.
    :param AOI: numpy array with the angles of incidence in degrees measured from the surface normal (90.0 = grazing,
                0.0 = normal incidence). The lengths of AOI and wavelengths have to match. Each entry has a one-to-one
                correspondence to the respective entry in the other array.
    :param wavelengths: numpy array with wavelengths in nm. The lengths of AOI and wavelengths have to match. Each entry
                        has a one-to-one correspondence to the respective entry in the other array.
    :param thicknesses: numpy array or list with the thicknesses of the layers in nm of a single period ordered from top
                        (vacuum side) to bottom (substrate side). Example: [0.5, 0.2] specifies two layers in a period
                        with thicknesses of 0.5 nm and 0.2 nm. In the stack, the 0.5 nm layer is above the 0.2 nm layer.
    :param compounds: list of strings specifying the materials of the layers given in the corresponding order as in
                      thicknesses. The lengths has to be identical to thicknesses. Example : ["Sc","Cr"] indicates that
                      the 0.5 nm layer defined in thicknesses is a scandium layer and that the 0.2 nm layer is a chromium
                      layer. The optical constants of the materials are automatically determined by the HenkeDataPD class
                      in the helper module. The strings have to match existing chemical elements or known compounds.
    :param periods: integer number of period replications in the multilayer system. The number defines how often the final
                    system stacks the layers defined in compounds and thicknesses.
    :param sigma: numpy array handed to the generate_mixlayers function: one dimensional numpy array with the interface
                  region thickness for the top interface of the respective layer. The length has to correspond to the
                  number of interfaces.
    :param substrate: string specifying the substrate material. Example "Si" for silicon.
    :param densities: numpy array of densities of the respective materials defined in compounds. The numbers given are
                      relative to the tabulated bulk density. Example: [0.9, 1.2] corresponds to 90 and 120 percent of
                      the bulk density of scandium and chromium for the example given in compounds. Has to match the
                      length of compounds and thicknesses.
    :param intermix: numpy array handed to the generate_mixlayers function: one dimensional numpy array with the
                     intermixing fraction of the previous layer with the next. The length has to correspond to the number
                     of interfaces.
    :param mixlayers: number of sampling points for the interface region. Handed to generate_mixlayers function.
    :param capthicknesses: numpy array or list of thicknesses of the capping layer ordered form vaccum side towards the
                           periodic stack side. Similar in structure to thicknesses, but specifies only the capping layers.
    :param cap: list of strings specifying the materials of the capping layers. Has to has the same lengths and order as
                capthicknesses.
    :param capdensities: list of density values relative to bulk density for the capping layers.
    :return: tuple (n,t) containing the optical constants and thicknesses for the other functions of this project.
    """
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
    """
    Calculate the reflected and transmitted field amplitudes at each interface for the specified experimental and model
    parameters. This is the core user function to apply the matrix method for a given multilayer system.
    :param AOI: numpy array with the angles of incidence in degrees measured from the surface normal (90.0 = grazing,
                0.0 = normal incidence). The lengths of AOI and wavelengths have to match. Each entry has a one-to-one
                correspondence to the respective entry in the other array.
    :param wavelengths: numpy array with wavelengths in nm. The lengths of AOI and wavelengths have to match. Each entry
                        has a one-to-one correspondence to the respective entry in the other array.
    :param thicknesses: numpy array or list with the thicknesses of the layers in nm of a single period ordered from top
                        (vacuum side) to bottom (substrate side). Example: [0.5, 0.2] specifies two layers in a period
                        with thicknesses of 0.5 nm and 0.2 nm. In the stack, the 0.5 nm layer is above the 0.2 nm layer.
    :param compounds: list of strings specifying the materials of the layers given in the corresponding order as in
                      thicknesses. The lengths has to be identical to thicknesses. Example : ["Sc","Cr"] indicates that
                      the 0.5 nm layer defined in thicknesses is a scandium layer and that the 0.2 nm layer is a chromium
                      layer. The optical constants of the materials are automatically determined by the HenkeDataPD class
                      in the helper module. The strings have to match existing chemical elements or known compounds.
    :param periods: integer number of period replications in the multilayer system. The number defines how often the final
                    system stacks the layers defined in compounds and thicknesses.
    :param sigma: numpy array handed to the generate_mixlayers function: one dimensional numpy array with the interface
                  region thickness for the top interface of the respective layer. The length has to correspond to the
                  number of interfaces.
    :param roughness: single double specifying the Nevot-Croce parameters for all interfaces.
    :param substrate: string specifying the substrate material. Example "Si" for silicon.
    :param densities: numpy array of densities of the respective materials defined in compounds. The numbers given are
                      relative to the tabulated bulk density. Example: [0.9, 1.2] corresponds to 90 and 120 percent of
                      the bulk density of scandium and chromium for the example given in compounds. Has to match the
                      length of compounds and thicknesses.
    :param intermix: numpy array handed to the generate_mixlayers function: one dimensional numpy array with the
                     intermixing fraction of the previous layer with the next. The length has to correspond to the number
                     of interfaces.
    :param mixlayers: number of sampling points for the interface region. Handed to generate_mixlayers function.
    :param capthicknesses: numpy array or list of thicknesses of the capping layer ordered form vaccum side towards the
                           periodic stack side. Similar in structure to thicknesses, but specifies only the capping layers.
    :param cap: list of strings specifying the materials of the capping layers. Has to has the same lengths and order as
                capthicknesses.
    :param capdensities: list of density values relative to bulk density for the capping layers.
    :return: tuple (refl, trans, n, t) with the reflected and transmitted complex field amplitudes in each layer and the layer
             system represented through the optical constants in n and the thicknesses of each layer in t. The reflectance
             measured from the sample can be calculated by abs(refl[0])**2.
    """
    n,t = generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities)
    
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    refl, trans = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t[1:], roughness)
    return refl, trans, n, t

def fields_dwba(AOI, AOE, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix,  mixlayers, capthicknesses, cap,capdensities):
    """
    Convinience function identical to the function fields. It calculates the field amplitudes required for the DWBA.
    It only differs from fields through the evaluation of the amplitudes for an additional set of angles specified in AOE.
    They represent the exit angles for the diffuse scattering experiment.
    :param AOE: numpy array of exit angles. Has to be identical in lengths to AOI.
    :return: tuple (r1, t1, r2, t2, qx, qz, t, n). The results r1 and t1 are identical to refl and trans in the return
             tuple of the function fields. The variables r2 and t2 correspond to the angles specified in AOE. In addition,
             the qx and qz calculations for use in the DWBA are returned.
    """
    n,t = generate_layer_system_matrix(AOI, wavelengths, thicknesses, compounds, periods, sigma, substrate, densities, intermix,  mixlayers, capthicknesses, cap, capdensities)
    
    kz_AOI, kx_AOI = k_z_generator(np.array(AOI),np.array(wavelengths),n)
    kz_AOE, kx_AOE = k_z_generator(np.array(AOE),np.array(wavelengths),n)
    r1, t1 = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOI), t[1:], roughness)
    r2, t2 = matrixmethod.amplitudes(n, np.array(wavelengths), np.array(kz_AOE), t[1:], roughness)
    
    qz = qz_gen(kz_AOI,kz_AOE)
    qx = kx_AOI - kx_AOE
    
    return r1, t1, r2, t2, qx, qz, t, n
    

def xrr(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness,  substrate, densities, intermix, mixlayers, capthicknesses, cap,capdensities):
    """
    Conviniece function to calculate only the reflectivity from a given system. Identical to fields, but only returns
    the numpy array with the reflectivity values.
    :return: reflectivity, corresponding to abs(refl[0])**2 in the function fields.
    """
    re, tr ,n, t = fields(AOI, wavelengths, thicknesses, compounds, periods, sigma, roughness, substrate, densities, intermix, mixlayers, capthicknesses, cap,capdensities)
    return np.abs(re[0])**2
    

    
    
    