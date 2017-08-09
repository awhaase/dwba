import numpy
import numba
@numba.jit(nopython=True)
def m11(k_z1, k_z2,z):
    return ((k_z2+k_z1)/(2*k_z1))*numpy.exp(-1j*z*k_z2)
@numba.jit(nopython=True)
def m12(k_z1, k_z2,z):
    return ((k_z1-k_z2)/(2*k_z1))*numpy.exp(+1j*z*k_z2)
@numba.jit(nopython=True)
def m21(k_z1, k_z2,z):
    return ((k_z1-k_z2)/(2*k_z1))*numpy.exp(-1j*z*k_z2)
@numba.jit(nopython=True)
def m22(k_z1, k_z2,z):
    return ((k_z2+k_z1)/(2*k_z1))*numpy.exp(+1j*z*k_z2)

def amplitudes(n,wavelengths,k_z_1,thickness, sigma):
    if numpy.isscalar(sigma):
        return amplitudes_single_sigma(n,wavelengths,k_z_1,thickness, sigma)
    else:
        return amplitudes_multiple_sigma(n,wavelengths,k_z_1,thickness, sigma)

@numba.jit(nopython=True)
def amplitudes_single_sigma(n,wavelengths,k_z_1,thickness, sigma):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    EP1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    
    EM1[-1,:] = 1 + 0j

    for w in xrange(len(wavelengths)):
        i = len(n)-2
        while (i >= 0):
            ROUGH1R = numpy.exp(-2*k_z_1[i,w]*k_z_1[i+1,w]*sigma**2)
            ROUGH1T = numpy.exp((k_z_1[i,w]-k_z_1[i+1,w])**2*0.5*sigma**2)

            M11K1 = m11(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*(1/ROUGH1T)
            M12K1 = m12(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*ROUGH1R*(1/ROUGH1T)
            M21K1 = m21(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*ROUGH1R*(1/ROUGH1T)
            M22K1 = m22(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*(1/ROUGH1T)
            EM1[i,w] = M11K1*EM1[i+1,w] + M12K1*EP1[i+1,w]
            EP1[i,w] = M21K1*EM1[i+1,w] + M22K1*EP1[i+1,w]
            i -=1
        
    t1 = EM1/EM1[0]
    r1 = EP1/EM1[0]
    
    return r1, t1

@numba.jit(nopython=True)
def amplitudes_multiple_sigma(n,wavelengths,k_z_1,thickness, sigma):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    EP1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    
    EM1[-1,:] = 1 + 0j

    for w in xrange(len(wavelengths)):
        i = len(n)-2
        while (i >= 0):
            ROUGH1R = numpy.exp(-2*k_z_1[i,w]*k_z_1[i+1,w]*sigma[i]**2)
            ROUGH1T = numpy.exp((k_z_1[i,w]-k_z_1[i+1,w])**2*0.5*sigma[i]**2)

            M11K1 = m11(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*(1/ROUGH1T)
            M12K1 = m12(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*ROUGH1R*(1/ROUGH1T)
            M21K1 = m21(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*ROUGH1R*(1/ROUGH1T)
            M22K1 = m22(k_z_1[i,w],k_z_1[i+1,w],thickness[i,0])*(1/ROUGH1T)
            EM1[i,w] = M11K1*EM1[i+1,w] + M12K1*EP1[i+1,w]
            EP1[i,w] = M21K1*EM1[i+1,w] + M22K1*EP1[i+1,w]
            i -=1
        
    t1 = EM1/EM1[0]
    r1 = EP1/EM1[0]
    
    return r1, t1
   
   
def amplitudes_bk(n,wavelengths,k_z_1,thickness, sigma):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    EP1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    
    EM1[-1,:] = 1 + 0j

    for w in xrange(len(wavelengths)):
        ROUGH1R = numpy.exp(-2*k_z_1[:-1,w]*k_z_1[1:,w]*sigma**2)
        ROUGH1T = numpy.exp((k_z_1[:-1,w]-k_z_1[1:,w])**2*0.5*sigma**2)

        M11K1 = m11(k_z_1[:,w],thickness[:,0])*(1/ROUGH1T)
        M12K1 = m12(k_z_1[:,w],thickness[:,0])*ROUGH1R*(1/ROUGH1T)
        M21K1 = m21(k_z_1[:,w],thickness[:,0])*ROUGH1R*(1/ROUGH1T)
        M22K1 = m22(k_z_1[:,w],thickness[:,0])*(1/ROUGH1T)

        i = len(n)-2
        while (i >= 0):
            EM1[i,w] = M11K1[i]*EM1[i+1,w] + M12K1[i]*EP1[i+1,w]
            EP1[i,w] = M21K1[i]*EM1[i+1,w] + M22K1[i]*EP1[i+1,w]
            i -=1
        
    t1 = EM1/EM1[0]
    r1 = EP1/EM1[0]
    
    return r1, t1

@numba.jit(nopython=True)
def m11_bk(k_z,z):
    return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])
@numba.jit(nopython=True)
def m12_bk(k_z,z):
    return ((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:])
@numba.jit(nopython=True)
def m21_bk(k_z,z):
    return ((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])
@numba.jit(nopython=True)
def m22_bk(k_z,z):
    return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:])
   