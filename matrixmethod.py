import numpy
import numba
@numba.jit(nopython=True)
def m11(k_z1, k_z2,z):
    """
    Private function. Calculates the M11 matrix element for the propagation matrix.
    :param k_z1: Above layer k_z array
    :param k_z2: Bottom layer k_z array
    :param z: thickness of the layer
    :return: M11 matrix element
    """
    return ((k_z2+k_z1)/(2*k_z1))*numpy.exp(-1j*z*k_z2)

@numba.jit(nopython=True)
def m12(k_z1, k_z2,z):
    """
    Private function. Calculates the M12 matrix element for the propagation matrix.
    :param k_z1: Above layer k_z array
    :param k_z2: Bottom layer k_z array
    :param z: thickness of the layer
    :return: M12 matrix element
    """
    return ((k_z1-k_z2)/(2*k_z1))*numpy.exp(+1j*z*k_z2)

@numba.jit(nopython=True)
def m21(k_z1, k_z2,z):
    """
    Private function. Calculates the M21 matrix element for the propagation matrix.
    :param k_z1: Above layer k_z array
    :param k_z2: Bottom layer k_z array
    :param z: thickness of the layer
    :return: M21 matrix element
    """
    return ((k_z1-k_z2)/(2*k_z1))*numpy.exp(-1j*z*k_z2)

@numba.jit(nopython=True)
def m22(k_z1, k_z2,z):
    """
    Private function. Calculates the M22 matrix element for the propagation matrix.
    :param k_z1: Above layer k_z array
    :param k_z2: Bottom layer k_z array
    :param z: thickness of the layer
    :return: M22 matrix element
    """
    return ((k_z2+k_z1)/(2*k_z1))*numpy.exp(+1j*z*k_z2)

def amplitudes(n,wavelengths,k_z_1,thickness, sigma):
    """
    Calculates the field amplitudes R and T at each interface of the multilayer for a single or multiple Nevot-Croce
    factors. The multilayer is defined through the optical constants matrix n and the incidence angle contained in k_z
    for the wavelengths given in wavelengths. The thicknesses are contained in thickness and the Nevot-Croce factor is
    given by the scalar or numpy array sigma.
    :param n: complex 2D numpy array with the shape (no of layers + vacuum and substrate, len(wavelengths)). The array
              contains the indices of refraction in the formal delta + i*beta. The first row n[0,:] of the array is the
              vacuum index of refraction, the last row n[-1,:] is the substrates index of refraction.
    :param wavelengths: numpy array with wavelengths in nm
    :param k_z: numpy array with the same shape as n. Result of the function helper.k_z_generator.
    :param thickness: numpy array with thicknesses of the layers in the stack. Has to be identical in lengths to the
                      shape of the first dimension of n (len(n[:,0]) minus one. The last element, corresponding to the
                      substrate thickness has to be set to 0.
    :param sigma: float indicating the Nevot-Croce factor OR numpy array indicating different Nevot-Croce fators for
                  each interface.
    :return: tuple of numpy arrays (R,T) with field complex field amplitudes at the top interfaces of the individual
             layers. The expression abs(R[0])**2 is thus the reflectivity measured in vacuum above the sample.
    """
    if numpy.isscalar(sigma):
        return amplitudes_single_sigma(n,wavelengths,k_z_1,thickness, sigma)
    else:
        return amplitudes_multiple_sigma(n,wavelengths,k_z_1,thickness, sigma)

@numba.jit(nopython=True)
def amplitudes_single_sigma(n, wavelengths, k_z, thickness, sigma):
    """
    Calculates the field amplitudes R and T at each interface of the multilayer for a single Nevot-Croce factor.
    The multilayer is defined through the optical constants matrix n and the incidence angle contained in k_z for the
    wavelengths given in wavelengths. The thicknesses are contained in thickness and the Nevot-Croce factor is given by
    the scalar sigma.
    :param n: complex 2D numpy array with the shape (no of layers + vacuum and substrate, len(wavelengths)). The array
              contains the indices of refraction in the formal delta + i*beta. The first row n[0,:] of the array is the
              vacuum index of refraction, the last row n[-1,:] is the substrates index of refraction.
    :param wavelengths: numpy array with wavelengths in nm
    :param k_z: numpy array with the same shape as n. Result of the function helper.k_z_generator.
    :param thickness: numpy array with thicknesses of the layers in the stack. Has to be identical in lengths to the
                      shape of the first dimension of n (len(n[:,0]) minus one. The last element, corresponding to the
                      substrate thickness has to be set to 0.
    :param sigma: float indicating the Nevot-Croce factor.
    :return: tuple of numpy arrays (R,T) with field complex field amplitudes at the top interfaces of the individual
             layers. The expression abs(R[0])**2 is thus the reflectivity measured in vacuum above the sample.
    """
    EP1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    
    EM1[-1,:] = 1 + 0j

    for w in xrange(len(wavelengths)):
        i = len(n)-2
        while (i >= 0):
            #This is where the roughness/intermixing with the Nevot-Croce factor enters. Different versions are possible.
            ROUGH1R = numpy.exp(-2 * k_z[i, w] * k_z[i + 1, w] * sigma ** 2)
            ROUGH1T = numpy.exp((k_z[i, w] - k_z[i + 1, w]) ** 2 * 0.5 * sigma ** 2)

            #calculate matrix elements
            M11K1 = m11(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * (1 / ROUGH1T)
            M12K1 = m12(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * ROUGH1R * (1 / ROUGH1T)
            M21K1 = m21(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * ROUGH1R * (1 / ROUGH1T)
            M22K1 = m22(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * (1 / ROUGH1T)

            #calculate field amplitudes
            EM1[i,w] = M11K1*EM1[i+1,w] + M12K1*EP1[i+1,w]
            EP1[i,w] = M21K1*EM1[i+1,w] + M22K1*EP1[i+1,w]
            i -=1

    #normalize field amplitudes
    T = EM1/EM1[0]
    R = EP1/EM1[0]
    
    return R, T

@numba.jit(nopython=True)
def amplitudes_multiple_sigma(n, wavelengths, k_z, thickness, sigma):
    """
    Calculates the field amplitudes R and T at each interface of the multilayer for multiple Nevot-Croce factors.
    The multilayer is defined through the optical constants matrix n and the incidence angle contained in k_z for the
    wavelengths given in wavelengths. The thicknesses are contained in thickness and the Nevot-Croce factors are given
    by the array sigma.
    :param n: complex 2D numpy array with the shape (no of layers + vacuum and substrate, len(wavelengths)). The array
              contains the indices of refraction in the formal delta + i*beta. The first row n[0,:] of the array is the
              vacuum index of refraction, the last row n[-1,:] is the substrates index of refraction.
    :param wavelengths: numpy array with wavelengths in nm
    :param k_z: numpy array with the same shape as n. Result of the function helper.k_z_generator.
    :param thickness: numpy array with thicknesses of the layers in the stack. Has to be identical in lengths to the
                      shape of the first dimension of n (len(n[:,0]) minus one. The last element, corresponding to the
                      substrate thickness has to be set to 0.
    :param sigma: numpy array with individual Nevot-Croce factors for each interface. The lengths has to correspond to
                  the number of interfaces in the system.
    :return: tuple of numpy arrays (R,T) with field complex field amplitudes at the top interfaces of the individual
             layers. The expression abs(R[0])**2 is thus the reflectivity measured in vacuum above the sample.
    """
    EP1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    
    EM1[-1,:] = 1 + 0j

    for w in xrange(len(wavelengths)):
        i = len(n)-2
        while (i >= 0):
            # This is where the roughness/intermixing with the Nevot-Croce factor enters. Different versions are possible.
            ROUGH1R = numpy.exp(-2 * k_z[i, w] * k_z[i + 1, w] * sigma[i] ** 2)
            ROUGH1T = numpy.exp((k_z[i, w] - k_z[i + 1, w]) ** 2 * 0.5 * sigma[i] ** 2)

            # calculate matrix elements
            M11K1 = m11(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * (1 / ROUGH1T)
            M12K1 = m12(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * ROUGH1R * (1 / ROUGH1T)
            M21K1 = m21(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * ROUGH1R * (1 / ROUGH1T)
            M22K1 = m22(k_z[i, w], k_z[i + 1, w], thickness[i, 0]) * (1 / ROUGH1T)

            # calculate field amplitudes
            EM1[i,w] = M11K1*EM1[i+1,w] + M12K1*EP1[i+1,w]
            EP1[i,w] = M21K1*EM1[i+1,w] + M22K1*EP1[i+1,w]
            i -=1

    # normalize field amplitudes
    T = EM1/EM1[0]
    R = EP1/EM1[0]
    
    return R, T