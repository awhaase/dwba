import numpy

def m11(k_z,z):
    return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])

def m12(k_z,z):
    return ((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:])

def m21(k_z,z):
    return ((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])

def m22(k_z,z):
    return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:])

def amplitudes(n,wavelengths,k_z_1,thickness, sigma):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    EP1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelengths))) + 0j
    
    EM1[-1,:] = 1 + 0j
    
    ROUGH1R = numpy.exp(-2*k_z_1[:-1]*k_z_1[1:]*sigma**2)
    ROUGH1T = numpy.exp((k_z_1[:-1]-k_z_1[1:])**2*0.5*sigma**2)
    
    M11K1 = m11(k_z_1,thickness)*(1/ROUGH1T)
    M12K1 = m12(k_z_1,thickness)*ROUGH1R*(1/ROUGH1T)
    M21K1 = m21(k_z_1,thickness)*ROUGH1R*(1/ROUGH1T)
    M22K1 = m22(k_z_1,thickness)*(1/ROUGH1T)

   
    for i in reversed(range(len(n)-1)):
        EM1[i,:] = M11K1[i,:]*EM1[i+1,:] + M12K1[i,:]*EP1[i+1,:]
        EP1[i,:] = M21K1[i,:]*EM1[i+1,:] + M22K1[i,:]*EP1[i+1,:]
        
    #t1 = EM1/EM1
    r1 = EP1/EM1
    
    return r1[0]