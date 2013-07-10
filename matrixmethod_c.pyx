import numpy
cimport numpy
#def m11(k_z,z):
    #return ((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:])

#def m12(k_z,z):
    #return (((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:]))

#def m21(k_z,z):
    #return (((k_z[:-1]-k_z[1:])/(2*k_z[:-1]))*numpy.exp(-1j*z*k_z[1:]))

#def m22(k_z,z):
    #return (((k_z[1:]+k_z[:-1])/(2*k_z[:-1]))*numpy.exp(+1j*z*k_z[1:]))
    
DTYPE = numpy.complex
ctypedef numpy.complex_t DTYPE_t

def amplitudes(numpy.ndarray n,numpy.ndarray wavelength,numpy.ndarray k_z_1,numpy.ndarray k_z_2,numpy.ndarray thickness):
    """
    Calculates the amplitudes for a rough multilayer in specular geometry for the wave vectors k_z_1 and k_z_2
    @param n numpy array optical constants
    @param wavelength numpy array with all wavelength at which the theory should be evaluated
    """
    cdef numpy.ndarray[DTYPE_t, ndim=2] EP1 = numpy.zeros((len(n),len(wavelength)), dtype=complex) + 0j
    cdef numpy.ndarray[DTYPE_t, ndim=2] EM1 = numpy.zeros((len(n),len(wavelength)), dtype=complex) + 0j
    cdef numpy.ndarray[DTYPE_t, ndim=2] EP2 = numpy.zeros((len(n),len(wavelength)), dtype=complex) + 0j
    cdef numpy.ndarray[DTYPE_t, ndim=2] EM2 = numpy.zeros((len(n),len(wavelength)), dtype=complex) + 0j
    
    EM1[len(n)-1,:] = 1 + 0j
    EM2[len(n)-1,:] = 1 + 0j
    
    #cdef numpy.ndarray ROUGH1R = numpy.exp(-2*k_z_1[:-1]*k_z_1[1:]*0.5**2)
    #cdef numpy.ndarray ROUGH1T = numpy.exp((k_z_1[:-1]-k_z_1[1:])**2*0.5*0.5**2)
    #cdef numpy.ndarray ROUGH2R = numpy.exp(-2*k_z_2[:-1]*k_z_2[1:]*0.5**2)
    #cdef numpy.ndarray ROUGH2T = numpy.exp((k_z_2[:-1]-k_z_2[1:])**2*0.5*0.5**2)
    
    cdef int i
    cdef int m
    
    
    for i in reversed(range(len(n)-1)):
        for m in range(len(wavelength)):
            print i
            print m
            #EM1[i,m] = ((k_z_1[i+1,m]+k_z_1[i,m])/(2*k_z_1[i,m]))*numpy.exp(-1j*thickness[i,m]*k_z_1[i+1,m])*EM1[i+1,m]+((k_z_1[i,m]-k_z_1[i+1,m])/(2*k_z_1[i,m]))*numpy.exp(+1j*thickness[i,m]*k_z_1[i+1,m])*EP1[i+1,m]
            EM1[i,m] =  k_z_1[i+1,m]
            #EP1[i,m] = (((k_z_1[i,m]-k_z_1[i+1,m])/(2*k_z_1[i,m]))*numpy.exp(-1j*thickness[i,m]*k_z_1[i+1,m]))*EM1[i+1,m]+(((k_z_1[i+1,m]+k_z_1[i,m])/(2*k_z_1[i,m]))*numpy.exp(+1j*thickness[i,m]*k_z_1[i+1,m]))*EP1[i+1,m]
            #EM2[i,m] = (((k_z_2[i+1,m]+k_z_2[i,m])/(2*k_z_2[i,m]))*numpy.exp(-1j*thickness[i,m]*k_z_2[i+1,m]))*EM2[i+1,m]+(((k_z_2[i,m]-k_z_2[i+1,m])/(2*k_z_2[i,m]))*numpy.exp(+1j*thickness[i,m]*k_z_2[i+1,m]))*EP2[i+1,m]
            #EP2[i,m] = (((k_z_2[i,m]-k_z_2[i+1,m])/(2*k_z_2[i,m]))*numpy.exp(-1j*thickness[i,m]*k_z_2[i+1,m]))*EM2[i+1,m]+(((k_z_2[i+1,m]+k_z_2[i,m])/(2*k_z_2[i,m]))*numpy.exp(+1j*thickness[i,m]*k_z_2[i+1,m]))*EP2[i+1,m]
    print "done"
    #cdef numpy.ndarray r1 = numpy.multiply((EP1[:-1,:]/EM1[:-1,:]),ROUGH1R)
    #cdef numpy.ndarray t1 = numpy.multiply((EM1[1:,:]/EM1[:-1,:]),ROUGH1T)
    #cdef numpy.ndarray r2 = numpy.multiply((EP2[:-1,:]/EM2[:-1,:]),ROUGH2R)
    #cdef numpy.ndarray t2 = numpy.multiply((EM2[1:,:]/EM2[:-1,:]),ROUGH2T)
    
    #cdef numpy.ndarray r1 = numpy.multiply((EP1[:-1,:]/EM1[:-1,:]),ROUGH1R)
    #cdef numpy.ndarray t1 = numpy.multiply((EM1[1:,:]/EM1[:-1,:]),ROUGH1T)
    #cdef numpy.ndarray r2 = numpy.multiply((EP2[:-1,:]/EM2[:-1,:]),ROUGH2R)
    #cdef numpy.ndarray t2 = numpy.multiply((EM2[1:,:]/EM2[:-1,:]),ROUGH2T)
    
    #return r1,t1,r2,t2,EP1,EM1,EP2,EM2
    return EP1,EM1,EP2,EM2