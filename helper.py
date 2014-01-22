import numpy as np

class HenkeData(object):
    
    def __init__(self, henkeDataFile):
        #print "Henke Daten lesen"
        import csv
        file = open(henkeDataFile)
        data = csv.reader(file, delimiter=' ',skipinitialspace=1)
        
        HenkeData = []
        for row in data:
            if data.line_num > 2:
                HenkeData.append([float(row[0]), float(row[1]), float(row[2])])
        
        self._data = np.array(HenkeData)
        
    def getDelta(self, wavelength):
        nearest_wl_index = (np.abs(wavelength - self._data[:,0])).argmin()
        return self._data[nearest_wl_index,1]
    
    def getBeta(self, wavelength):
        nearest_wl_index = (np.abs(wavelength - self._data[:,0])).argmin()
        return self._data[nearest_wl_index,2]

def angle(theta,n_up,n_low):
    return np.arcsin(n_up*np.sin(theta)/n_low)

def snell(k_z_vac, k_x_vac, n):
    return np.sqrt(n**2*(k_z_vac**2+k_x_vac**2) -(k_x_vac**2))

def fresnel_r_s(n_i,theta_i,n_j,theta_j):
    return (n_i*np.cos(theta_i) - n_j*np.cos(theta_j))/(n_i*np.cos(theta_i)+n_j*np.cos(theta_j))

def fresnel_t_s(n_i,theta_i,n_j,theta_j):
    return (2*n_i*np.cos(theta_i))/(n_i*np.cos(theta_i)+n_j*np.cos(theta_j))

def k_z_generator(angle,ls, n):
    k_z_vac = 2*np.pi*np.cos(np.radians(angle))/np.array(ls)
    k_x_vac = 2*np.pi*np.sin(np.radians(angle))/np.array(ls)
    k_z = snell(k_z_vac,k_x_vac,np.array(n))
    return k_z, k_x_vac
    
    
def qz_gen(k_z_1,k_z_2):
    q0z = -(k_z_2 + k_z_1)
    q1z = -(k_z_1 - k_z_2)
    q2z = -q1z
    q3z = -q0z
    return (q0z,q1z,q2z,q3z)