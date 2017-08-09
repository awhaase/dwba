import sys
import glob
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.pyplot.style.use(['bmh','white_background'])
from scipy.constants import c,pi
from scipy.constants import physical_constants
sys.path.append(u'/home/ahaase/pc/Daten/HDF++/')
import bessyhdfviewer as hdf
import bessyhdfviewer
import csv
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import leastsq, curve_fit
from scipy import optimize
import copy
from scipy.constants import c,pi
from scipy.constants import physical_constants
from scipy.optimize import curve_fit, leastsq
h_planck = physical_constants['Planck constant in eV s'][0]

class Parameter:
    def __init__(self, value):
        self.value = value
    
    def set(self, value):
        self.value = value
    
    def __call__(self):
        return self.value
    
class BigRef(object):
    
    _rawdata = {}
    _data = {}
    _interpolation = {}
    _datapoints = {}
    _HDFFileList =  []
    _qx_min = 0.0
    _qx_max = 0.0
    _qz_min = 0.0
    _qz_max = 0.0
    
    #some standard definitions
    
    def __init__(self, HDFFiles):
        self._HDFFileList = []
        for hdf in HDFFiles:
            self._HDFFileList.append(glob.glob(hdf))
        self._HDFFileList = [file for sublist in self._HDFFileList for file in sublist]
    
    def _calcQ(self, AOI, TwoTheta, wl):
        """
        Calculate the q coordinates for given image array and experimental
        parameters.
        """

        K0 = 2*pi*np.ones(np.shape(wl))/wl
            
        af   = np.radians(TwoTheta - AOI)
        ai   = np.radians(AOI)
        th_f = 0.0
        
        #print af
        
       
        real_af = af
        real_ai = ai
        
        #determine q coordinates for all pixels
        qx = K0 * ( np.cos(th_f) * np.sin(real_af) - np.sin(real_ai) )
        qy = K0 * ( np.sin(th_f) * np.sin(real_af) )
        qz = K0 * ( np.cos(real_af) + np.cos(real_ai) )
        
        return (qx, qy, qz, real_af, real_ai)
        

    def _drawLines(self, wavelength, angleOut, angleIn, angleR=15, sim=False):
        if sim:
            plt.hlines(0.931, self._qx_min,self._qx_max,colors='g',lw=40)
        for w in wavelength:
            line_qx = []
            line_qz = []
            
            for i in xrange(np.shape(angleOut)[0]):
                #wli = np.nanargmin(np.abs(w - np.array(data[str(theta)]['Wavelength'])))
                #real_ai = data[str(theta)]['real_ai'][wli]
                #real_af = data[str(theta)]['real_af'][wli]
                #af   = real_af
                #ai   = real_ai
                
                af   = np.radians(90.0 - angleOut[i])
                ai   = np.radians(90.0 - angleIn[i])
                ar   = np.radians(90-angleR)
                lineqx = 2*np.pi/((w/np.sin(ar))*np.sin(ai))*(np.cos(af) - np.cos(ai))
                lineqz = 2*np.pi/((w/np.sin(ar))*np.sin(ai))*(np.sin(af) + np.sin(ai))
                line_qx.append(lineqx)
                line_qz.append(lineqz)
                
            plt.plot(line_qx,line_qz,'b-', lw=1)  
        
        ### Show dynamical scattering for wavelength specified above 
        for w in wavelength:
            line_qx = []
            line_qz = []
            
            for i in xrange(np.shape(angleOut)[0]):
                #wli = np.nanargmin(np.abs(w - np.array(data[str(theta)]['Wavelength'])))
                #real_ai = data[str(theta)]['real_ai'][wli]
                #real_af = data[str(theta)]['real_af'][wli]
                #af   = real_af
                #ai   = real_ai
                af   = np.radians(90.0 - angleOut[i])
                ai   = np.radians(90.0 - angleIn[i])
                ar   = np.radians(90-angleR)
                lineqx = 2*np.pi/((w/np.sin(ar))*np.sin(af))*(np.cos(af) - np.cos(ai))
                lineqz = 2*np.pi/((w/np.sin(ar))*np.sin(af))*(np.sin(af) + np.sin(ai))
                line_qx.append(lineqx)
                line_qz.append(lineqz)
                
            plt.plot(line_qx,line_qz,'r-', lw=1)

            
    def _showDataInQ(self, wavelength, angleIn, angleOut, angleR=6.75, show=True,log=True,threeD=False):
        #global data, rawdata, interpolation, qx_min, qx_max, qz_min, qz_max, q_dim_x, q_dim_z

        imgArrayQSpace = self._interpolation['data']
        
        #if threeD and log:
            #mlab.surf(np.log(imgArrayQSpace), warp_scale=50)
        #elif threeD and not log:
            #mlab.surf(imgArrayQSpace)
        
    
        #plt.clf()
        fig = None
        plot = None
        ax = None
        if log:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot = ax.imshow(np.log(imgArrayQSpace), extent=(self._qx_min,self._qx_max,self._qz_min,self._qz_max), aspect=2.5)
            #plot.set_clim(-15,-9.5 )
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot = ax.imshow(imgArrayQSpace, extent=(self._qx_min,self._qx_max,self._qz_min,self._qz_max), aspect=2.5, cmap="jet")
            plot.set_clim([0,1E-3])
            #cb = colorbar(plot, format='%.1e')
            #cb.set_label("Intensity $I/I_0$")
            #plot.set_clim(-15,-10)
        ### Show real data points
        #for angle in self._data.keys():
        #    plt.plot(self._data[angle]['qx'],self._data[angle]['qz'], 'r,')
        #    plt.annotate(self._data[angle]['qx'][-1],(self._data[angle]['qx'][-1],self._data[angle]['qz'][-1]))

        ### draw lines for specified wavelength
        self._drawLines(wavelength, angleOut, angleIn, angleR)
        
        
        ### Set-up the plot
        
        #plot.set_cmap('jet')
        #cmap = plt.get_cmap('jet')
        #cmap.set_bad((0.0,0.0,0.5))
        
        #plot.set_clim(-15,-10)
        
        ax.set_xlabel('$q_x$ [1/nm]')
        ax.set_ylabel('$q_z$ [1/nm]')
        #plot.ylim(0.875,1.0)
        #plot.xlim(-0.2,0.2)
        #plt.title('GISAXS PTB#17, %s eV' % energy)
        #if show:
        #    plt.show()
        
        #SI EDGE
        
        qx = -2*np.pi/12.5*(np.sin(np.radians(angleIn))-np.sin(np.radians(angleOut)))
        qz = 2*np.pi/12.5*(np.cos(np.radians(angleIn))+np.cos(np.radians(angleOut)))
        
        qx2 = -2*np.pi/12.35*(np.sin(np.radians(angleIn))-np.sin(np.radians(angleOut)))
        qz2 = 2*np.pi/12.35*(np.cos(np.radians(angleIn))+np.cos(np.radians(angleOut)))
        
        qx3 = -2*np.pi/14.00*(np.sin(np.radians(angleIn))-np.sin(np.radians(angleOut)))
        qz3 = 2*np.pi/14.00*(np.cos(np.radians(angleIn))+np.cos(np.radians(angleOut)))
        
        #ax.plot(qx,qz,'b--',lw=1)
        #ax.plot(qx2,qz2,'r-',lw=1)
        #ax.plot(qx3,qz3,'r-',lw=1)
        return fig, ax, plot

        
        ### LINE PROFILE EXTRACTION ###
    def _gridpointForQ(self, qx, qz):
        qx_grid = np.nanargmin(np.abs(qx - self._interpolation['qxgrid'][0,:]))
        qz_grid = np.nanargmin(np.abs(qz - self._interpolation['qzgrid'][:,0]))
        
        return (qx_grid, qz_grid)
    
class ElliResonantReflectivitySX700(BigRef):
    
    def __init__(self, HDFFiles, TwoTheta, normDataFile, avrg=3, skip=0,offset=0, darkcurrent=0, error=0):
        super(ElliResonantReflectivitySX700, self).__init__(HDFFiles)
        self._TwoTheta = TwoTheta
        self._normDataFile = normDataFile
        self._avrg = avrg
        self._offset = offset
        self._darkcurrent = darkcurrent
        self._error = error
        self._skip = skip
        self._noninterpolated = []
        
    def showDataInQ(self, wavelength, ar, DeltaT):
        angleIn = [a for a in np.sort(np.array([float(a) for a in self._data.keys()]))]
        angleOut= [DeltaT-a for a in np.sort(np.array([float(a) for a in self._data.keys()]))]
        return self._showDataInQ(wavelength, angleIn, angleOut,ar, True, False)
        
    def process(self, 
                badFiles=[], AOIcorrection=0.0, TwoThetaCorrection=0.0,
                energy_min=-0.5, energy_max=0.5,
                qz_min=2.6, qz_max=2.9,
                q_dim_x=200, q_dim_z=1600):
                    
        self._energy_min=energy_min
        self._energy_max=energy_max
        self._qz_min=qz_min
        self._qz_max=qz_max
        self._q_dim_x=q_dim_x
        self._q_dim_z=q_dim_z
        
        self._data = {}
        self._interpolation = {}
       
        normHDFData = hdf.bessy_reshape(self._normDataFile)
        norm = normHDFData['Detector']['normVal01']['data']
        norm = norm[self._offset:]
        self._norm = norm

        print "started processing..."
        iter = 0
        for HDFfile in self._HDFFileList:
            #print HDFfile
            hdfdata = hdf.bessy_reshape(HDFfile)
            AOI     = hdfdata['MotorPositions']['Theta-S']+AOIcorrection
            #Phi     = hdfdata['MotorPositions']['Phi']
            wl       = hdfdata['Motor']['SX700-Energy']['data']
            MeasuredData = hdfdata['Detector']['normVal01']['data']
            #Filter = hdfdata['MotorPositions']['HIOS-Filter']
            
            #print '\nprocessing: ', HDFfile
            #print 'wl: ', np.round(wl[0],2)
            iter += 1
        
            #drop bad files
            if HDFfile in badFiles:
                continue
                                
            #qx, qy, qz, real_af, real_ai = self._calcQ(AOI, self._TwoTheta+TwoThetaCorrection, h_planck*c/(1E-9*np.array(wl)))
            
            qz = 2*np.pi/(h_planck*c/(1E-9*np.array(wl)))*2*np.cos(np.radians(AOI))
            qx=0.0
            qy=0.0
            
            daten = []
            error = []
            daten = [(np.mean(MeasuredData[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg])-self._darkcurrent)/np.mean(norm[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
            error = [(np.std(MeasuredData[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg])+self._error)/np.mean(norm[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]

            self._rawdata.update({"{0}".format(str(np.round(AOI,3))) : {'AOI': AOI, 'Wavelength': wl,  'Data' : MeasuredData, 'qz' : qz}})
            self._data.update({"{0}".format(str(np.round(AOI,3))) : {'AOI' : AOI,'AOF' : self._TwoTheta-AOI, 'Wavelength' : wl[::self._avrg], 'Data' : daten, 'Error': error, 'qz' : qz[::self._avrg]}})
    
        #do the interpolation
        print "finished processing %i files!" % iter
        self._datapoints = np.array([])
        energy = np.array([])
        qz = np.array([])    
        
        for AOI in sorted(self._data.keys(), key=float):
            self._datapoints = np.concatenate((self._datapoints,self._data[AOI]['Data']))
            self._noninterpolated.append(self._data[AOI]['Data'])
            energy = np.concatenate((energy,self._data[AOI]['Wavelength']))
            qz = np.concatenate((qz,self._data[AOI]['qz']))
        
        qz_coord, energy_coord = np.mgrid[qz_max:qz_min:complex(0,q_dim_z), 1e-2*energy_min:1e-2*energy_max:complex(0,q_dim_x)]
        points =[[1e-2*energy[i],qz[i]] for i in xrange(np.shape(energy)[0])]
        
        imgArrayQSpace = griddata(points, self._datapoints, (energy_coord, qz_coord), method='linear')
        
        #imgArrayQSpace = matplotlib.mlab.griddata(qx, qz, imgArrayPoints, qx_coord, qz_coord)
        self._interpolation.update({'data' : imgArrayQSpace, 'energygrid' : energy_coord, 'qzgrid' : qz_coord})
        self._noninterpolated = np.array(self._noninterpolated)
        
    def qx_profile(self, qx):
        return
    
    def qz_profile(self, qz, qx_min=-0.4, qx_max=0.0):
        profile = []
        aoi = []
        aof = []
        l = []
        qx = []
        qza = []
        for key in self._data.keys():
            idx = np.abs(self._data[key]["qz"]-qz).argmin()
            if (self._data[key]["qx"][idx]>=qx_min and self._data[key]["qx"][idx]<=qx_max):
                qx.append(self._data[key]["qx"][idx])
                profile.append(self._data[key]["Data"][idx])
                aoi.append(self._data[key]["AOI"])
                aof.append(self._data[key]["AOF"])
                l.append(self._data[key]["Wavelength"][idx])
                qza.append(self._data[key]["qz"][idx])
        return (np.array(aoi),np.array(aof),np.array(l),np.array(qx),np.array(qza),np.array(profile))
        
        
class ElliRockingSX700(BigRef):
    
    def __init__(self, HDFFiles, TwoTheta, normDataFile, avrg=3, skip=0,offset=0, darkcurrent=0, error=0):
        super(ElliRockingSX700, self).__init__(HDFFiles)
        self._TwoTheta = TwoTheta
        self._normDataFile = normDataFile
        self._avrg = avrg
        self._offset = offset
        self._darkcurrent = darkcurrent
        self._error = error
        self._skip = skip
        
    def showDataInQ(self, wavelength, ar, DeltaT):
        angleIn = [a for a in np.sort(np.array([float(a) for a in self._data.keys()]))]
        angleOut= [DeltaT-a for a in np.sort(np.array([float(a) for a in self._data.keys()]))]
        return self._showDataInQ(wavelength, angleIn, angleOut,ar, True, False)
        
    def process(self, 
                badFiles=[], AOIcorrection=0.0, TwoThetaCorrection=0.0,
                qx_min=-0.5, qx_max=0.5,
                qz_min=2.6, qz_max=2.9,
                q_dim_x=1600, q_dim_z=1600):
                    
        self._qx_min=qx_min
        self._qx_max=qx_max
        self._qz_min=qz_min
        self._qz_max=qz_max
        self._q_dim_x=q_dim_x
        self._q_dim_z=q_dim_z
        
        self._data = {}
        self._interpolation = {}
       
        normHDFData = hdf.bessy_reshape(self._normDataFile)
        norm = normHDFData['Detector']['normVal01']['data']
        norm = norm[self._offset:]
        self._norm = norm

        print "started processing..."
        iter = 0
        for HDFfile in self._HDFFileList:
            #print HDFfile
            hdfdata = hdf.bessy_reshape(HDFfile)
            AOI     = hdfdata['MotorPositions']['Theta-S']+AOIcorrection
            #Phi     = hdfdata['MotorPositions']['Phi']
            wl       = hdfdata['Motor']['SX700-Wavelength']['data']
            MeasuredData = hdfdata['Detector']['normVal01']['data']
            #Filter = hdfdata['MotorPositions']['HIOS-Filter']
            
            #print '\nprocessing: ', HDFfile
            #print 'wl: ', np.round(wl[0],2)
            iter += 1
        
            #drop bad files
            if HDFfile in badFiles:
                continue
                                
            qx, qy, qz, real_af, real_ai = self._calcQ(AOI, self._TwoTheta+TwoThetaCorrection, wl)
            
            ##OPTIONAL
            #qx = -qx
            
            daten = []
            error = []
            daten = [(np.mean(MeasuredData[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg])-self._darkcurrent)/np.mean(norm[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
            error = [(np.std(MeasuredData[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg])+self._error)/np.mean(norm[(self._avrg*i+self._skip):(self._avrg*i)+self._avrg]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]

            self._rawdata.update({"{0}".format(str(np.round(AOI,3))) : {'AOI': AOI, 'Wavelength': wl,  'Data' : MeasuredData, 'qx' : qx, 'qy' : qy, 'qz' : qz}})
            self._data.update({"{0}".format(str(np.round(AOI,3))) : {'AOI' : AOI,'AOF' : self._TwoTheta-AOI, 'Wavelength' : wl[::self._avrg], 'Data' : daten, 'Error': error, 'qx' : qx[::self._avrg], 'qy' : qy[::self._avrg], 'qz' : qz[::self._avrg]}})
    
        #do the interpolation
        print "finished processing %i files!" % iter
        self._datapoints = np.array([])
        qx = np.array([])
        qz = np.array([])    
        
        for AOI in self._data.keys():
            self._datapoints = np.concatenate((self._datapoints,self._data[AOI]['Data']))
            qx = np.concatenate((qx,self._data[AOI]['qx']))
            qz = np.concatenate((qz,self._data[AOI]['qz']))
        
        qz_coord, qx_coord = np.mgrid[qz_max:qz_min:complex(0,q_dim_z), qx_min:qx_max:complex(0,q_dim_x)]
        points =[[qx[i],qz[i]] for i in xrange(np.shape(qx)[0])]
        
        imgArrayQSpace = griddata(points, self._datapoints, (qx_coord, qz_coord), method='linear')
        
        #imgArrayQSpace = matplotlib.mlab.griddata(qx, qz, imgArrayPoints, qx_coord, qz_coord)
        self._interpolation.update({'data' : imgArrayQSpace, 'qxgrid' : qx_coord, 'qzgrid' : qz_coord})
        
    def qx_profile(self, qx):
        return
    
    def qz_profile(self, qz, qx_min=-0.4, qx_max=0.0):
        profile = []
        aoi = []
        aof = []
        l = []
        qx = []
        qza = []
        for key in self._data.keys():
            idx = np.abs(self._data[key]["qz"]-qz).argmin()
            if (self._data[key]["qx"][idx]>=qx_min and self._data[key]["qx"][idx]<=qx_max):
                qx.append(self._data[key]["qx"][idx])
                profile.append(self._data[key]["Data"][idx])
                aoi.append(self._data[key]["AOI"])
                aof.append(self._data[key]["AOF"])
                l.append(self._data[key]["Wavelength"][idx])
                qza.append(self._data[key]["qz"][idx])
        return (np.array(aoi),np.array(aof),np.array(l),np.array(qx),np.array(qza),np.array(profile))
        
        
        
def cutgenerator_rear(scan, keys):
    qzcuts = []
    for key in keys:
        qzcuts.append([scan._data[key]['AOI'],
                       scan._data[key]['AOF'],
                       np.array(scan._data[key]['Wavelength']),
                       np.array(scan._data[key]['Data'])-np.mean(np.array(scan._data[key]['Data'])[:5]),
                       np.ones(len(np.array(scan._data[key]['Data'])))*np.nanmax(np.array(scan._data[key]['Data']))*0.05])
    return qzcuts   

def cutgenerator(scan, keys):
    qzcuts = []
    for key in keys:
        qzcuts.append([scan._data[key]['AOI'],
                       scan._data[key]['AOF'],
                       np.array(scan._data[key]['Wavelength']),
                       np.array(scan._data[key]['Data'])-np.mean(np.array(scan._data[key]['Data'])[:-5:-1]),
                       np.ones(len(np.array(scan._data[key]['Data'])))*np.nanmax(np.array(scan._data[key]['Data']))*0.02])
    return qzcuts

class KalphaXRR:
    def __init__(self, filename, lim=" "):
        #print "Henke Daten lesen"
        import csv
        with open(filename) as file:
            data = csv.reader(file, delimiter=lim,skipinitialspace=1)
            
            XRRData = []
            for row in data:
                if data.line_num > 2:
                    XRRData.append([float(row[0]), float(row[1])])
            
            self._dataraw = np.array(XRRData)
            maxv = np.nanmax(self._dataraw[:,1])
            self._data = np.zeros((self._dataraw.shape[0],3))
            self._data[:,0]=self._dataraw[:,0]
            self._data[:,1]=self._dataraw[:,1]/maxv
            self._data[:,2]=np.sqrt(self._dataraw[:,1])/maxv
        
        
def processHDF(badFiles,HDFFiles, AOIcorrection, avrg):
        
        HDFFileList = []
        for hdf in HDFFiles:
            HDFFileList.append(glob.glob(hdf))
        HDFFileList = [file for sublist in HDFFileList for file in sublist]
        
        
      
        #print "processing..."
        iter = 0
        data = np.array([])
        qx = np.array([])
        qz = np.array([])   
        
        for HDFfile in HDFFileList:
            print HDFfile
            hdfdata = bessyhdfviewer.bessy_reshape(HDFfile)
            AOI     = np.array(hdfdata['Motor']['Theta']['data'])+AOIcorrection
            MeasuredData = hdfdata['Detector']['K1norm']['data']
            
            iter += 1
        
            #drop bad files
            if HDFfile in badFiles:
                print "skipped"
                continue
         
            daten = np.array(MeasuredData)
            if len(data)>1:
                data = append((data,np.array(daten)))
            else:
                data = np.array(daten)

        return AOI[AOI>-1E10], data[AOI>-1E10]
    
# REUV functions:
def lorentzian(x, x0, gamma, a, background):
    return (a/pi)*0.5*gamma/((x-x0)**2+(0.5*gamma)**2)+background

def process(badFiles,HDFFiles, AOIcorrection, avrg):
        
        HDFFileList = []
        for hdf in HDFFiles:
            HDFFileList.append(glob.glob(hdf))
        HDFFileList = [file for sublist in HDFFileList for file in sublist]
        
        
      
        #print "processing..."
        iter = 0
        data = np.array([])
        qx = np.array([])
        qz = np.array([])   
        
        for HDFfile in HDFFileList:
            print HDFfile
            hdfdata = bessyhdfviewer.bessy_reshape(HDFfile)
            AOI     = np.array(hdfdata['Motor']['Theta-S']['data'])+AOIcorrection
            energy       = hdfdata['Motor']['SX700-Energy']['data']
            MeasuredData = hdfdata['Detector']['normVal01']['data']
            
            iter += 1
        
            #drop bad files
            if HDFfile in badFiles:
                print "skipped"
                continue
         
            daten = np.array(MeasuredData)
            if len(data)>1:
                data = append((data,np.array(daten)))
            else:
                data = np.array(daten)

        return data, AOI, energy

def batch(HDFFiles, NormHDFFiles):
        d = np.array([])
        a = np.array([])
        e = np.array([])
        HDFFileList = []
        for hdf in HDFFiles:
            HDFFileList.append(glob.glob(hdf))
        HDFFileList = [file for sublist in HDFFileList for file in sublist]
        for file in HDFFileList:
            data, aoi, energy = process([],[file],0,1)
            d = np.concatenate((d, data))
            a = np.concatenate((a, aoi))
            e = np.concatenate((e, energy))
        
        #norm
        d_n = np.array([])
        a_n = np.array([])
        e_n = np.array([])
        NormHDFFileList = []
        for hdf_n in NormHDFFiles:
            NormHDFFileList.append(glob.glob(hdf_n))
        NormHDFFileList = [file for sublist in NormHDFFileList for file in sublist]
        for file in NormHDFFileList:
            data, aoi, energy = process([],[file],0,1)
            d_n = np.concatenate((d_n, data))
            a_n = np.concatenate((a_n, aoi))
            e_n = np.concatenate((e_n, energy))
        return d,a,e, e_n, d_n
    
    
    
    
    
    
def normalize(a, d, e, d_n, e_n, scale=1.0):
    step = np.mean(np.abs(np.array(e_n)[:-1]-np.array(e_n)[1:]))
    print step
    energies = np.arange(min(e),max(e)+step,step)
    dataarray = []
    anglearray = []
    energyarray = []
    for en in energies:
        idx = np.where(np.array([True if (x>=(en-step/4) and x<=(en+step/4)) else False for x in e]))
        idx_norm = np.where(np.array([True if (x>=(en-step/4) and x<=(en+step/4)) else False for x in e_n]))
        if len(idx_norm)>1 or len(idx_norm)<1:
            print "error!"
        else:
            dataarray.append(np.array(d[idx])/(scale*d_n[idx_norm]))
            anglearray.append(np.array(a[idx]))
            energyarray.append(np.array(e[idx]))
    return dataarray, anglearray, energies, energyarray
    
    
    
    
    
    
    
def grid(title, data, aoi, energy, energy_norm, data_norm, res=1000):
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from matplotlib.colors import LogNorm
    qz = 2*np.pi*energy/(h_planck*c)*2*np.cos(np.radians(aoi))*1e-9
    aoi_c, energy_c =  np.mgrid[np.nanmin(aoi):np.nanmax(aoi):complex(0,res), np.nanmin(energy):np.nanmax(energy):complex(0,res)]
    qz_c, energy_c =  np.mgrid[np.nanmin(qz):np.nanmax(qz):complex(0,res), np.nanmin(energy):np.nanmax(energy):complex(0,res)]
    fig = figure()
    
    res = griddata((aoi,energy),data,(aoi_c,energy_c), method="linear")
    
    ax = fig.add_subplot(111, title=title)
    #img = ax.contourf(energy_c,aoi_c, log10(res),300, cmap='hot')
    img = ax.imshow(log(res), extent=(np.nanmin(energy_c), np.nanmax(energy_c),np.nanmin(aoi_c), np.nanmax(aoi_c)), origin='lower', cmap='gray')
    #colorbar(img)
    #img.set_clim([0.0,0.15])
    fig.canvas.draw()
    #plot(energy, aoi, 'r,')
    
    #res = griddata((qz,energy),data,(qz_c,energy_c), method="nearest")
    
    #ax = fig.add_subplot(212)
    #ax.imshow(log(res), extent=(nanmin(energy_c), nanmax(energy_c),nanmin(qz_c), nanmax(qz_c)), origin='lower', aspect=100)
    #plot(energy, qz, 'r,')
    return res, aoi_c, energy_c

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def fitting_pos_width(e_min, e_max, e_step, energy, angle, data, plot=False):
    energies = arange(e_min,e_max,e_step)
    dataarray = []
    anglearray = []
    for en in energies:
        idx = where(array([True if (x>=(en-0.1) and x<=(en+0.1)) else False for x in energy]))
        dataarray.append(array(data[idx]))
        anglearray.append(array(angle[idx]))
        
    centerpos = []
    halfwidth = []
    for i in xrange(len(dataarray)):
        maxval_idx = argmax(dataarray[i])
        maxval = array(dataarray[i])[maxval_idx]
        maxvalpos = array(anglearray[i])[maxval_idx]
        popt, pcov = curve_fit(lorentzian,anglearray[i],dataarray[i],p0=[maxvalpos, 1.0, maxval,0.0])

        centerpos.append(popt[0])
        halfwidth.append(popt[1])
        
        if plot:
            figure()
            plot(anglearray[i],dataarray[i], 'ro')
            plot(anglearray[i], lorentzian(anglearray[i],*popt), color='b')
            vlines(popt[0],0,nanmax(dataarray[i]), color='b')
            hlines(1/(pi*popt[1])*popt[2]+popt[3],popt[0]-0.5*popt[1],popt[0]+0.5*popt[1], color='b')
    return (energies, centerpos, halfwidth, anglearray, dataarray)

def fitting_pos_width_energies(energies, energy, angle, data, showplot=False):
    #energies = arange(e_min,e_max,e_step)
    dataarray = []
    anglearray = []
    for en in energies:
        idx = where(array([True if (x>=(en-0.1) and x<=(en+0.1)) else False for x in energy]))
        dataarray.append(array(data[idx]))
        anglearray.append(array(angle[idx]))
        
    centerpos = []
    halfwidth = []
    for i in xrange(len(dataarray)):
        maxval_idx = argmax(dataarray[i])
        maxval = array(dataarray[i])[maxval_idx]
        maxvalpos = array(anglearray[i])[maxval_idx]
        popt, pcov = curve_fit(lorentzian,anglearray[i],dataarray[i],p0=[maxvalpos, 1.0, maxval,0.0], maxfev=10000)

        centerpos.append(popt[0])
        halfwidth.append(popt[1])
        
        if showplot:
            figure()
            plot(anglearray[i],dataarray[i], 'ro')
            plot(anglearray[i], lorentzian(anglearray[i],*popt), color='b')
            vlines(popt[0],0,nanmax(dataarray[i]), color='b')
            hlines(1/(pi*popt[1])*popt[2]+popt[3],popt[0]-0.5*popt[1],popt[0]+0.5*popt[1], color='b')
    return (energies, centerpos, halfwidth)

def nm2eV(wl):
    return h_planck*c/(wl*1E-9)