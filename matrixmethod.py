import numpy

def m11(k_z,z):
    return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])

def m12(k_z,z):
    return ((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:])

def m21(k_z,z):
    return ((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])

def m22(k_z,z):
    return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:])

def amplitudes(n,wavelength,k_z_1,k_z_2,thickness, sigma):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    
    EP1 = numpy.zeros((len(n),len(wavelength))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelength))) + 0j
    EP2 = numpy.zeros((len(n),len(wavelength))) + 0j
    EM2 = numpy.zeros((len(n),len(wavelength))) + 0j
    
    EM1[-1,:] = 1 + 0j
    EM2[-1,:] = 1 + 0j
    
    
    M11K1 = m11(k_z_1,thickness)
    M12K1 = m12(k_z_1,thickness)
    M21K1 = m21(k_z_1,thickness)
    M22K1 = m22(k_z_1,thickness)
    M11K2 = m11(k_z_2,thickness)
    M12K2 = m12(k_z_2,thickness)
    M21K2 = m21(k_z_2,thickness)
    M22K2 = m22(k_z_2,thickness)
   
    for i in reversed(range(len(n)-1)):
        EM1[i,:] = M11K1[i,:]*EM1[i+1,:] + M12K1[i,:]*EP1[i+1,:]
        EP1[i,:] = M21K1[i,:]*EM1[i+1,:] + M22K1[i,:]*EP1[i+1,:]
        EM2[i,:] = M11K2[i,:]*EM2[i+1,:] + M12K2[i,:]*EP2[i+1,:]
        EP2[i,:] = M21K2[i,:]*EM2[i+1,:] + M22K2[i,:]*EP2[i+1,:]
        
    t1 = EM1/EM1[0]
    t2 = EM2/EM2[0]
    r1 = EP1/EM1[0]
    r2 = EP2/EM2[0]
    
    return r1[:],r2[:],t1[:],t2[:]

def amplitudes_debeyewaller(n,wavelength,k_z_1,k_z_2,thickness, sigma):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    
    EP1 = numpy.zeros((len(n),len(wavelength))) + 0j
    EM1 = numpy.zeros((len(n),len(wavelength))) + 0j
    EP2 = numpy.zeros((len(n),len(wavelength))) + 0j
    EM2 = numpy.zeros((len(n),len(wavelength))) + 0j
    
    EM1[-1,:] = 1 + 0j
    EM2[-1,:] = 1 + 0j
    
    ROUGH1R = numpy.exp(-2*k_z_1[:-1]*k_z_1[1:]*sigma**2)
    ROUGH1T = numpy.exp((k_z_1[:-1]-k_z_1[1:])**2*0.5*sigma**2)
    #ROUGH1T = 1
    ROUGH2R = numpy.exp(-2*k_z_2[:-1]*k_z_2[1:]*sigma**2)
    ROUGH2T = numpy.exp((k_z_2[:-1]-k_z_2[1:])**2*0.5*sigma**2)
    #ROUGH2T = 1
    
    M11K1 = m11(k_z_1,thickness)*(1/ROUGH1T)
    M12K1 = m12(k_z_1,thickness)*ROUGH1R*(1/ROUGH1T)
    M21K1 = m21(k_z_1,thickness)*ROUGH1R*(1/ROUGH1T)
    M22K1 = m22(k_z_1,thickness)*(1/ROUGH1T)
    M11K2 = m11(k_z_2,thickness)*(1/ROUGH2T)
    M12K2 = m12(k_z_2,thickness)*ROUGH2R*(1/ROUGH2T)
    M21K2 = m21(k_z_2,thickness)*ROUGH2R*(1/ROUGH2T)
    M22K2 = m22(k_z_2,thickness)*(1/ROUGH2T)
   
    for i in reversed(range(len(n)-1)):
        EM1[i,:] = M11K1[i,:]*EM1[i+1,:] + M12K1[i,:]*EP1[i+1,:]
        EP1[i,:] = M21K1[i,:]*EM1[i+1,:] + M22K1[i,:]*EP1[i+1,:]
        EM2[i,:] = M11K2[i,:]*EM2[i+1,:] + M12K2[i,:]*EP2[i+1,:]
        EP2[i,:] = M21K2[i,:]*EM2[i+1,:] + M22K2[i,:]*EP2[i+1,:]
        
    t1 = EM1/EM1[0]
    t2 = EM2/EM2[0]
    r1 = EP1/EM1[0]
    r2 = EP2/EM2[0]
    
    return r1[:-1],r2[:-1],t1[:-1],t2[:-1]