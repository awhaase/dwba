import glob
import sys
import numpy as np
sys.path.append(u'/home/ahaase/pc/Daten/HDF++/')
import bessyhdfviewer as hdf

from scipy.constants import c,pi
from scipy.constants import physical_constants
from scipy.interpolate import griddata
import copy

import fitting
import reflectivity

class BigRef(object):
    
    _rawdata = {}
    _data = {}
    _interpolation = {}
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
    
    def _calcQ(self, AOI, TwoTheta, wl, refracDelta=None):
        """
        Calculate the q coordinates for given image array and experimental
        parameters.
        """

        K0 = 2*pi*np.ones(np.shape(wl))/wl
            
        af   = np.radians(90.0 - (TwoTheta - AOI))
        ai   = np.radians(90.0 - AOI)
        th_f = 0.0
        
        #print af
        
        if refracDelta:
            real_af = np.arcsin(np.sqrt(np.sin(af)**2 - [2*d for d in refracDelta] + [d**2 for d in refracDelta]))
            real_ai = np.arcsin(np.sqrt(np.sin(ai)**2 - [2*d for d in refracDelta] + [d**2 for d in refracDelta]))
        else:
            real_af = af
            real_ai = ai

        #print 'real af:', 90-real_af[::3]*180/np.pi
        #print 'real_ai:', 90-real_ai[::3]*180/np.pi
        
        #determine q coordinates for all pixels
        qx = K0 * ( np.cos(th_f) * np.cos(real_af) - np.cos(real_ai) )
        qy = K0 * ( np.sin(th_f) * np.cos(real_af) )
        qz = K0 * ( np.sin(real_af) + np.sin(real_ai) )
        
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
            fig = figure()
            ax = fig.add_subplot(111)
            plot = ax.imshow(np.log(imgArrayQSpace), extent=(self._qx_min,self._qx_max,self._qz_min,self._qz_max))
            plot.set_clim(-15,-9.5 )
        else:
            fig = figure()
            ax = fig.add_subplot(111)
            plot = ax.imshow(imgArrayQSpace, extent=(self._qx_min,self._qx_max,self._qz_min,self._qz_max), aspect=2.5, cmap="gray_r")
            cb = colorbar(plot, format='%.1e')
            cb.set_label("Intensity $I/I_0$")
            #plot.set_clim(-15,-10)
        ### Show real data points
        #for angle in self._data.keys():
        #    plt.plot(self._data[angle]['qx'],self._data[angle]['qz'], 'r,')
        #    plt.annotate(self._data[angle]['qx'][-1],(self._data[angle]['qx'][-1],self._data[angle]['qz'][-1]))

        ### draw lines for specified wavelength
        self._drawLines(wavelength, angleOut, angleIn, angleR)
        
        
        ### Set-up the plot
        
        plot.set_cmap('jet')
        cmap = plt.get_cmap('jet')
        cmap.set_bad((0.0,0.0,0.5))
        
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
        
        ax.plot(qx,qz,'b--',lw=1)
        ax.plot(qx2,qz2,'r-',lw=1)
        ax.plot(qx3,qz3,'r-',lw=1)
        return fig, ax, plot

        
        ### LINE PROFILE EXTRACTION ###
    def _gridpointForQ(self, qx, qz):
        qx_grid = np.nanargmin(np.abs(qx - self._interpolation['qxgrid'][0,:]))
        qz_grid = np.nanargmin(np.abs(qz - self._interpolation['qzgrid'][:,0]))
        
        return (qx_grid, qz_grid)

    def _AveragedProfile(self, pos, direction, avrgPx, imgArray):
        """
        Returns an array containing the averaged profile along either the
        x or y direction for specified pixel position
        """
        
        dimx, dimz = np.shape(imgArray)
        
        if direction == "qx":
            size = dimz
        else:
            size = dimx
            
        meanProfile = np.zeros(size, dtype=float)

        for p in xrange(0, size):
            if direction=="qx":
                meanProfile[p] = np.mean(imgArray[(pos-avrgPx):(pos+avrgPx),p])
            else:
                meanProfile[p] = np.mean(imgArray[p,(pos-avrgPx):(pos+avrgPx)])
        
        return meanProfile

    def _IntegratedProfile(self, pos, direction, intPx, imgArray):
        """
        Returns an array containing the averaged profile along either the
        x or y direction for specified pixel position
        """
        
        dimx, dimx = np.shape(imgArray)
        
        if direction == "qx":
            size = dimx
        else:
            size = dimx
            
        intProfile = np.zeros(size, dtype=float)
        
        for p in xrange(0, size):
            if direction=="qx":
                intProfile[p] = np.sum(imgArray[(pos-intPx):(pos+intPx),p])
            else:
                intProfile[p] = np.sum(imgArray[p,(pos-intPx):(pos+intPx)])
        
        return intProfile
        
    def IntegratedProfileAtQZOriginalData(self, qz, d_qz):
        """
        Profile on original data given any qz value.
        """
        profile = np.zeros(np.shape(self._data.keys())[0], dtype=float)
        qxpos = np.zeros(np.shape(self._data.keys())[0], dtype=float)
        
        for i in xrange(0,np.shape(self._data.keys())[0]):
            qzi = np.nanargmin(np.abs(qz - self._data[self._data.keys()[i]]['qz']))
            d_qzi_u = np.nanargmin(np.abs((qz+d_qz) - self._data[self._data.keys()[i]]['qz']))
            d_qzi_l = np.nanargmin(np.abs((qz-d_qz) - self._data[self._data.keys()[i]]['qz']))
            #print "limits: ", d_qzi_l, d_qzi_u
            profile[i] = np.mean(self._data[self._data.keys()[i]]['Data'][d_qzi_l:d_qzi_u])
            qxpos[i] = self._data[self._data.keys()[i]]['qx'][qzi]
        
        return qxpos, profile
    
    def IntegratedProfileAtQZOriginalDataWithError(self, qz, d_qz):
        """
        Profile on original data given any qz value.
        """
        profile = np.zeros(np.shape(self._data.keys())[0], dtype=float)
        error = np.zeros(np.shape(self._data.keys())[0], dtype=float)
        qxpos = np.zeros(np.shape(self._data.keys())[0], dtype=float)
        
        for i in xrange(0,np.shape(self._data.keys())[0]):
            qzi = np.nanargmin(np.abs(qz - self._data[self._data.keys()[i]]['qz']))
            d_qzi_u = np.nanargmin(np.abs((qz+d_qz) - self._data[self._data.keys()[i]]['qz']))
            d_qzi_l = np.nanargmin(np.abs((qz-d_qz) - self._data[self._data.keys()[i]]['qz']))
            #print self._data.keys()[i]
            #print "limits: ", d_qzi_l, d_qzi_u
            profile[i] = np.mean(self._data[self._data.keys()[i]]['Data'][d_qzi_l:d_qzi_u])
            error[i] = np.mean(self._data[self._data.keys()[i]]['Error'][d_qzi_l:d_qzi_u])
            qxpos[i] = self._data[self._data.keys()[i]]['qx'][qzi]
        
        return qxpos, profile, error

    def plotProfileInQ(self, direction, qx, qz, avrgQ, scale='log', show=False, preview=False, integrate=False):
        
        #global interpolation, qx_max, qx_min, qz_max, qz_min
        
        position = 'qz'
        avrgPx = 0
        qx_grid, qz_grid = self._gridpointForQ(qx,qz)

        if direction == 'qx':
            position = qz_grid
            qx_grid, posavrgPx  = self._gridpointForQ(qx, qz+avrgQ)
            avrgPx = np.abs(posavrgPx - position)
        else:
            position = qx_grid
            posavrgPx, qz_grid  = self._gridpointForQ(qx+avrgQ, qz)
            avrgPx = np.abs(posavrgPx - position)
            
        profile = None
        if integrate:
            profile = self._IntegratedProfile(position, direction, avrgPx, self._interpolation['data'])
        else:
            profile = self._AveragedProfile(position, direction, avrgPx, self._interpolation['data'])
        
        if preview:
            if scale=='log':
                self._showDataInQ([],[],[],6.75,False, True)
            else:
                self._showDataInQ([],[],[],6.75,False, False)
            if direction == 'qx':
                plt.hlines(qz, self._qx_min,self._qx_max, colors='r')
                plt.hlines(qz+avrgQ, self._qx_min,self._qx_max, colors='r', linestyles='dashed')
                plt.hlines(qz-avrgQ, self._qx_min,self._qx_max, colors='r', linestyles='dashed')
            elif direction == 'qz':
                plt.vlines(qx, self._qz_min,self._qz_max, colors='r')
                plt.vlines(qx+avrgQ, self._qz_min,self._qz_max, colors='r', linestyles='dashed')
                plt.vlines(qx-avrgQ, self._qz_min,self._qz_max, colors='r', linestyles='dashed')
            plt.show()
        
        qrange = []
        
        plthandle = None
        
        if direction == 'qx':
            plthandle = plt.plot(self._interpolation["qxgrid"][position,:], profile, 'ro-')
            qrange = self._interpolation["qxgrid"][position,:]
        else:
            plthandle = plt.plot(self._interpolation["qzgrid"][:,position], profile)
            qrange = self._interpolation["qzgrid"][:,position]
        #plt.yscale(scale)
        if show:
            plt.show()
            
        return qrange, profile
        
    def profileOriginalData(self, angleKey):
        plt.plot(self._data[angleKey]['Wavelength'], self._data[angleKey]['Data'], 'o')
        return (self._data[angleKey]['Wavelength'], self._data[angleKey]['Data'])
    
def cutgenerator(scan, keys):
    qzcuts = []
    for key in keys:
        qzcuts.append([scan._data[key]['AOI'],
                       scan._data[key]['AOF'],
                       np.array(scan._data[key]['Wavelength']),
                       np.array(scan._data[key]['Data']),
                       np.array(scan._data[key]['Error'])])
    return qzcuts

class BigRefRockingPTB17(BigRef):
    
    def __init__(self, HDFFiles, TwoTheta, normDataFile, avrg=3, offset=0, darkcurrent=0, error=0, skip=0):
        super(BigRefRockingPTB17, self).__init__(HDFFiles)
        self._TwoTheta = TwoTheta
        self._normDataFile = normDataFile
        self._avrg = avrg
        self._offset = offset
        self._skip = skip
        self._darkcurrent = darkcurrent
        self._error = error
        
    def showDataInQ(self, lines=False, show=True, log=False):
        angleIn = [a for a in np.sort(np.array([float(a) for a in self._data.keys()]))]
        angleOut= [29.94-a for a in np.sort(np.array([float(a) for a in self._data.keys()]))]
        if lines:
            wavelength = [12.92,13.06,13.76,13.94]
            #wavelength = [12.737,12.594,12.468,13.404]
        else:
            wavelength=[]
        #print angleIn
        #print self._data[self._data.keys()[0]]
        #print angleOut
        return self._showDataInQ(wavelength, angleIn, angleOut,6.75, show, log)
        
    def process(self, 
                badFiles=['daten/107_00070.hdf'], AOIcorrection=0.0, TwoThetaCorrection=0.0,
                qx_min=-0.4, qx_max=0.4,
                qz_min=0.8, qz_max=1.0,
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
        norm = normHDFData['Detector']['Keithley_1']['data']
        normRing = normHDFData['Detector']['MonDiode']['data']
        norm = norm[self._offset:]
        self._norm = norm

        print "started processing..."
        print self._HDFFileList
        iter = 0
        for HDFfile in self._HDFFileList:
            #print HDFfile
            hdfdata = hdf.bessy_reshape(HDFfile)
            AOI     = hdfdata['MotorPositions']['Theta']+AOIcorrection
            Phi     = hdfdata['MotorPositions']['Phi']
            wl       = hdfdata['Motor']['SX100Wavelength']['data']
            MeasuredData = hdfdata['Detector']['Keithley_1']['data']
            MeasuredRingCurrent = hdfdata['Detector']['MonDiode']['data']
            #Filter = hdfdata['MotorPositions']['HIOS-Filter']
            
            #print '\nprocessing: ', HDFfile
            #print 'wl: ', np.round(wl[0],2)
            iter += 1
        
            #drop bad files
            if HDFfile in badFiles:
                print "skipping %s" % HDFfile
                continue
                                
            qx, qy, qz, real_af, real_ai = self._calcQ(AOI, self._TwoTheta+TwoThetaCorrection, wl, None)
            
            daten = []
            error = []
            daten = [(np.mean((np.array(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg][self._skip:]))/np.array(MeasuredRingCurrent[(self._avrg*i):(self._avrg*i)+self._avrg][self._skip:]))-self._darkcurrent)/np.mean(np.array(norm[(self._avrg*i):(self._avrg*i)+self._avrg][self._skip:])/np.array(normRing[(self._avrg*i):(self._avrg*i)+self._avrg][self._skip:])) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
            error = [(np.std(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg][self._skip:])+self._error)/np.mean(norm[(self._avrg*i):(self._avrg*i)+self._avrg][self._skip:]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]

            self._rawdata.update({"{0}".format(str(np.round(AOI,2))) : {'AOI': AOI, 'Phi' : Phi, 'Wavelength': wl,  'Data' : MeasuredData, 'qx' : qx, 'qy' : qy, 'qz' : qz}})
            self._data.update({"{0}".format(str(np.round(AOI,2))) : {'AOI' : AOI,'AOF' : self._TwoTheta++TwoThetaCorrection-AOI, 'Phi' : Phi, 'Wavelength' : wl[::self._avrg], 'Data' : daten, 'Error': error, 'qx' : qx[::self._avrg], 'qy' : qy[::self._avrg], 'qz' : qz[::self._avrg]}})
    
        #do the interpolation
        print "finished processing %i files!" % iter
        imgArrayPoints = np.array([])
        qx = np.array([])
        qz = np.array([])    
        
        for AOI in self._data.keys():
            imgArrayPoints = np.concatenate((imgArrayPoints,self._data[AOI]['Data']))
            qx = np.concatenate((qx,self._data[AOI]['qx']))
            qz = np.concatenate((qz,self._data[AOI]['qz']))
        
        qz_coord, qx_coord = np.mgrid[qz_max:qz_min:complex(0,q_dim_z), qx_min:qx_max:complex(0,q_dim_x)]
        points =[[qx[i],qz[i]] for i in xrange(np.shape(qx)[0])]
        
        imgArrayQSpace = griddata(points, imgArrayPoints, (qx_coord, qz_coord), method='linear')
        
        #imgArrayQSpace = matplotlib.mlab.griddata(qx, qz, imgArrayPoints, qx_coord, qz_coord)
        self._interpolation.update({'data' : copy.deepcopy(imgArrayQSpace), 'qxgrid' : qx_coord, 'qzgrid' : qz_coord})
        
class BigRefRockingPTB17SX700(BigRef):
    
    def __init__(self, HDFFiles, TwoTheta, normDataFile, avrg=3, offset=0, darkcurrent=0, error=0):
        super(BigRefRockingPTB17SX700, self).__init__(HDFFiles)
        self._TwoTheta = TwoTheta
        self._normDataFile = normDataFile
        self._avrg = avrg
        self._offset = offset
        self._darkcurrent = darkcurrent
        self._error = error
        
    def showDataInQ(self, lines=False, show=True, log=False):
        angleIn = [a for a in np.sort(np.array([float(a)+2.891 for a in self._data.keys()]))]
        angleOut= [30-a for a in np.sort(np.array([float(a)+2.891 for a in self._data.keys()]))]
        if lines:
            wavelength = [12.92,13.06,13.76,13.94]
            #wavelength = [12.737,12.594,12.468,13.404]
        else:
            wavelength=[]
        return self._showDataInQ(wavelength, angleIn, angleOut,6.75, show, log)
        
    def process(self, 
                badFiles=['daten/107_00070.hdf'],
                qx_min=-0.3, qx_max=0.3,
                qz_min=0.85, qz_max=1.0,
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
            AOI     = hdfdata['MotorPositions']['Theta']
            Phi     = hdfdata['MotorPositions']['Phi']
            wl       = hdfdata['Motor']['SX700-Wavelength']['data']
            MeasuredData = hdfdata['Detector']['normVal01']['data']
            Filter = hdfdata['MotorPositions']['HIOS-Filter']
            
            #print '\nprocessing: ', HDFfile
            #print 'wl: ', np.round(wl[0],2)
            iter += 1
        
            #drop bad files
            if HDFfile in badFiles:
                continue
                                
            qx, qy, qz, real_af, real_ai = self._calcQ(AOI+2.891, self._TwoTheta, wl, None)
            
            daten = []
            error = []
            daten = [(np.mean(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])-self._darkcurrent)/np.mean(norm[(self._avrg*i):(self._avrg*i)+self._avrg]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
            error = [(np.std(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])+self._error)/np.mean(norm[(self._avrg*i):(self._avrg*i)+self._avrg]) for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]

            self._rawdata.update({"{0}".format(str(np.round(AOI,1))) : {'AOI': AOI, 'Phi' : Phi, 'Wavelength': wl,  'Data' : MeasuredData, 'qx' : qx, 'qy' : qy, 'qz' : qz}})
            self._data.update({"{0}".format(str(np.round(AOI,1))) : {'AOI' : AOI,'AOF' : self._TwoTheta-AOI,'Phi' : Phi, 'Wavelength' : wl[::self._avrg], 'Data' : daten, 'Error': error, 'qx' : qx[::self._avrg], 'qy' : qy[::self._avrg], 'qz' : qz[::self._avrg]}})
    
        #do the interpolation
        print "finished processing %i files!" % iter
        imgArrayPoints = np.array([])
        qx = np.array([])
        qz = np.array([])    
        
        for AOI in self._data.keys():
            imgArrayPoints = np.concatenate((imgArrayPoints,self._data[AOI]['Data']))
            qx = np.concatenate((qx,self._data[AOI]['qx']))
            qz = np.concatenate((qz,self._data[AOI]['qz']))
        
        qz_coord, qx_coord = np.mgrid[qz_max:qz_min:complex(0,q_dim_z), qx_min:qx_max:complex(0,q_dim_x)]
        points =[[qx[i],qz[i]] for i in xrange(np.shape(qx)[0])]
        
        imgArrayQSpace = griddata(points, imgArrayPoints, (qx_coord, qz_coord), method='cubic')
        
        #imgArrayQSpace = matplotlib.mlab.griddata(qx, qz, imgArrayPoints, qx_coord, qz_coord)
        self._interpolation.update({'data' : copy.deepcopy(imgArrayQSpace), 'qxgrid' : qx_coord, 'qzgrid' : qz_coord})
        
class BigRefTwoThetaPTB17SX700(BigRef):
    
    def __init__(self, HDFFiles, AOI, normDataFileFilterChange, normDataFileNoFilterChange, avrg, darkcurrent1=0, darkcurrent2=0, error=0):
        BigRef.__init__(self, HDFFiles)
        self._AOI = AOI
        self._normDataFileFilterChange = normDataFileFilterChange
        self._normDataFileNoFilterChange = normDataFileNoFilterChange
        self._avrg = avrg
        self._darkcurrent1 = darkcurrent1
        self._darkcurrent2 = darkcurrent2
        self._error = error
        
    def showDataInQ(self, lines=True, show=True, log=False):
        angleOut = np.sort(np.array([float(a)-self._AOI for a in self._data.keys()]))
        angleIn = [6.75 for i in angleOut]
        if lines:
            wavelength = [12.92,13.06,13.76,13.94]
        else:
            wavelength = []
            
        return self._showDataInQ(wavelength, angleIn, angleOut, 6.75, show, log)
        
    def process(self,temperr=0 ,
                badFiles=['daten/066.hdf','daten/066_00025.hdf','daten/066_00024.hdf','daten/066_00023.hdf','daten/066_00022.hdf','daten/066_00021.hdf','daten/066_00020.hdf', 'daten/066_00019.hdf'],
                qx_min=-0.3, qx_max=0.3,
                qz_min=0.85, qz_max=1.0,
                q_dim_x=1600, q_dim_z=1600):
                    
        self._qx_min=qx_min
        self._qx_max=qx_max
        self._qz_min=qz_min
        self._qz_max=qz_max
        self._q_dim_x=q_dim_x
        self._q_dim_z=q_dim_z
      
        normHDFDataFilterChange = hdf.bessy_reshape(self._normDataFileFilterChange)
        normFilterChange = normHDFDataFilterChange['Detector']['normVal01']['data']
        
        normHDFDataNoFilterChange = hdf.bessy_reshape(self._normDataFileNoFilterChange)
        normNoFilterChange = normHDFDataNoFilterChange['Detector']['normVal01']['data']
        print 'started processing...'
        iter = 0
        for HDFfile in self._HDFFileList:
            
            hdfdata = hdf.bessy_reshape(HDFfile)
            TwoTheta     = hdfdata['MotorPositions']['2Theta']
            wl       = hdfdata['Motor']['SX700-Wavelength']['data']
            MeasuredData = hdfdata['Detector']['normVal01']['data']
            Filter = hdfdata['MotorPositions']['HIOS-Filter']
            
            #print '\nprocessing: ', HDFfile
            iter += 1
            
        
            #drop bad files
            if HDFfile in badFiles:
                continue
                
            qx, qy, qz, real_af, real_ai = self._calcQ(self._AOI, TwoTheta+temperr, wl, None)
            
            daten = []
            MeasurementError = []
            if Filter > 25:
                if np.round(wl[0],2)==14.0:
                    daten = [(np.mean(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])-self._darkcurrent2)/normNoFilterChange[i] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
                    MeasurementError = [(np.std(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])+self._error)/normNoFilterChange[i] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
                else:
                    daten = [(np.mean(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])-self._darkcurrent1)/normNoFilterChange[i+50] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
                    MeasurementError = [(np.std(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])+self._error)/normNoFilterChange[i+50] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
            else:
                if np.round(wl[0],2)==14.0:
                    daten = [(np.mean(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])-self._darkcurrent2)/normFilterChange[i] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
                    MeasurementError = [(np.std(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])+self._error)/normFilterChange[i] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
                else:
                    daten = [(np.mean(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])-self._darkcurrent1)/normFilterChange[i+50] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
                    MeasurementError = [(np.std(MeasuredData[(self._avrg*i):(self._avrg*i)+self._avrg])+self._error)/normFilterChange[i+50] for i in xrange(np.shape(MeasuredData[::self._avrg])[0])]
           
            self._rawdata.update({"{0}".format(str(np.round(TwoTheta,1))) : {'TwoTheta': TwoTheta, 'Wavelength': wl,  'Data' : MeasuredData, 'Error' : MeasurementError, 'qx' : qx, 'qy' : qy, 'qz' : qz}})
            self._data.update({"{0}".format(str(np.round(TwoTheta,1))) : {'TwoTheta' : TwoTheta, 'Wavelength' : wl[::self._avrg], 'Data' : daten, 'Error' : MeasurementError, 'qx' : qx[::self._avrg], 'qy' : qy[::self._avrg], 'qz' : qz[::self._avrg]}})
    
        #do the interpolation
        print "finished processing %i files!" % iter
        imgArrayPoints = np.array([])
        qx = np.array([])
        qz = np.array([])    
             
        for TwoTheta in self._data.keys():
            imgArrayPoints = np.concatenate((imgArrayPoints,self._data[TwoTheta]['Data']))
            qx = np.concatenate((qx,self._data[TwoTheta]['qx']))
            qz = np.concatenate((qz,self._data[TwoTheta]['qz']))
        
        qz_coord, qx_coord = np.mgrid[qz_max:qz_min:complex(0,q_dim_z), qx_min:qx_max:complex(0,q_dim_x)]
        points =[[qx[i],qz[i]] for i in xrange(np.shape(qx)[0])]

        imgArrayQSpace = griddata(points, imgArrayPoints, (qx_coord, qz_coord), method='cubic')
        self._interpolation.update({'data' : imgArrayQSpace, 'qxgrid' : qx_coord, 'qzgrid' : qz_coord})