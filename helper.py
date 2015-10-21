import numpy as np


from scipy.constants import c,pi
from scipy.constants import physical_constants
h_planck = physical_constants['Planck constant in eV s'][0]
import periodictable as pt
import periodictable.xsf as xsf

def nm2eV(wl):
       return h_planck*c/(np.array(wl)*1E-9)
   
def plot_multilayer(axis, t, n, label=''):
    axis.step([np.sum(t[:i]) for i in xrange(len(t))], np.real(n[:]),where='post', label=label)
    
def plot_multilayer_imag(axis, t, n, label=''):
    axis.step([np.sum(t[:i]) for i in xrange(len(t))], np.imag(n[:]),where='post', label=label)
  
henke_densities = [
    ['', 'AgBr', 6.473],
    ['', 'AlAs', 3.81],
    ['', 'AlN', 3.26],
    ['Sapphire', 'Al2O3', 3.97],
    ['', 'AlP', 2.42],
    ['', 'B4C', 2.52],
    ['', 'BeO', 3.01],
    ['', 'BN', 2.25],
    ['Polyimide', 'C22H10N2O5', 1.43],
    ['Polypropylene', 'C3H6', 0.90],
    ['PMMA', 'C5H8O2', 1.19],
    ['Polycarbonate', 'C16H14O3', 1.2],
    ['Kimfol', 'C16H14O3', 1.2],
    ['Mylar', 'C10H8O4', 1.4],
    ['Teflon', 'C2F4', 2.2],
    ['Parylene-C', 'C8H7Cl', 1.29],
    ['Parylene-N', 'C8H8', 1.11],
    ['Fluorite', 'CaF2', 3.18],
    ['', 'CdWO4', 7.9],
    ['', 'CdS', 4.826],
    ['', 'CoSi2', 5.3],
    ['', 'Cr2O3', 5.21],
    ['', 'CsI', 4.51],
    ['', 'CuI', 5.63],
    ['', 'InN', 6.88],
    ['', 'In2O3', 7.179],
    ['', 'InSb', 5.775],
    ['', 'IrO2', 11.66],
    ['', 'GaAs', 5.316],
    ['', 'GaN', 6.10],
    ['', 'GaP', 4.13],
    ['', 'HfO2', 9.68],
    ['', 'LiF', 2.635],
    ['', 'LiH', 0.783],
    ['', 'LiOH', 1.43],
    ['', 'MgF2', 3.18],
    ['', 'MgO', 3.58],
    ['', 'Mg2Si', 1.94],
    ['Mica', 'KAl3Si3O12H2', 2.83],
    ['', 'MnO', 5.44],
    ['', 'MnO2', 5.03],
    ['', 'MoO2', 6.47],
    ['', 'MoO3', 4.69],
    ['', 'MoSi2', 6.31],
    ['Salt', 'NaCl', 2.165],
    ['', 'NbSi2', 5.37],
    ['', 'NbN', 8.47],
    ['', 'NiO', 6.67],
    ['', 'Ni2Si', 7.2],
    ['', 'Ru2Si3', 6.96],
    ['', 'RuO2', 6.97],
    ['', 'SiC', 3.217],
    ['', 'Si3N4', 3.44],
    ['Silica', 'SiO2', 2.2],
    ['Quartz', 'SiO2', 2.65],
    ['', 'TaN', 16.3],
    ['', 'TiN', 5.22],
    ['', 'Ta2Si', 14.],
    ['Rutile', 'TiO2', 4.26],
    ['ULE', 'Si.925Ti.075O2', 2.205],
    ['', 'UO2', 10.96],
    ['', 'VN', 6.13],
    ['Water', 'H2O', 1.0],
    ['', 'WC', 15.63],
    ['YAG', 'Y3Al5O12', 4.55],
    ['Zerodur', 'Si.56Al.5P.16Li.04Ti.02Zr.02Zn.03O2.46', 2.53],
    ['', 'ZnO', 5.675],
    ['', 'ZnS', 4.079],
    ['', 'ZrN', 7.09],
    ['Zirconia', 'ZrO2', 5.68],
    ['', 'ZrSi2', 4.88],
]  

def compound_density(compound, desperate_lookup=False):
    """Returns the density of the compound in g/cm^3. Elemental densities are taken from periodictable, which gets
    the densities from "The ILL Neutron Data Booklet, Second Edition."
    For compound densities, the values from the henke database at http://henke.lbl.gov/cgi-bin/density.pl are used
    if available.
    If the compound density is not found for the given compound, None is returned, unless desperate_lookup is True,
    in which case the elemental density of the first element in the compound is returned.
    """
    for d in henke_densities:
        if compound in (d[0], d[1]):
            return d[2]
    comp = pt.formula(compound)
    if comp.density is not None:
        return comp.density
    if desperate_lookup:
        return comp.structure[0][1].density
    return None

class HenkeData(object):
    
    def __init__(self, henkeDataFile):
        #print "Henke Daten lesen"
        import csv
        with open(henkeDataFile) as file:
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
    
class HenkeDataPD(object):
    
    def __init__(self, compound, wavelength): 
        if compound is "vac":
            self.n = np.ones(len(wavelength))
        else:
            density = compound_density(compound, True)
            wl = 1E1*np.array(wavelength) #nm to angstrom
            f1,f2 = xsf.xray_sld(compound ,wavelength=wl, density=density)
            n = np.conj(1 - wl**2/(2*np.pi)*(f1 + f2*1j)*1e-6)
            self.n = np.array(n)
        
    def getDelta(self):
        return np.real(1-self.n)
    
    def getBeta(self):
        return np.imag(self.n)
    
class HenkeDataAFF(object):
    
    def __init__(self, henkeDataFile):
        #print "Henke Daten lesen"
        import csv
        with open(henkeDataFile) as file:
            data = csv.reader(file, delimiter="\t",skipinitialspace=1)
            
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