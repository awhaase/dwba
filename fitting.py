from scipy.optimize import leastsq, curve_fit
import numpy as np
import reflectivity
import dwba
reload(dwba)
from helper import *
from multiprocessing import Pool
from ipyparallel import Client
import time
import matplotlib as plt


def residual_spec(angle_in, wl_spec, y_spec, stack, henke, period, s, densities,capthicknesses, caphenkes, capdensities):
    r = reflectivity.xrr(angle_in,
                     wl_spec,
                     stack,
                     henke,
                     period,
                     None, s,
                     "Si", densities, None, 0, capthicknesses, caphenkes, capdensities)
     

    err_spec = np.abs(y_spec-r)/(0.003*np.nanmax(y_spec))
    err_spec = err_spec**2/len(err_spec)

    return err_spec

def residual_spec_log(angle_in, wl_spec, y_spec, stack, henke, period, s, densities,capthicknesses, caphenkes, capdensities,clip_min, xrr_err):
    r = reflectivity.xrr(angle_in,
                     wl_spec,
                     stack,
                     henke,
                     period,
                     None, s,
                     "Si", densities, None, 0, capthicknesses, caphenkes, capdensities)
     

    err_spec = np.abs((np.log10(y_spec)-np.log10(np.clip(r,clip_min,5.0))))/np.abs(np.log10(y_spec+xrr_err)-np.log10(np.clip(y_spec-xrr_err,clip_min,1.0)))
    err_spec = err_spec**2/len(err_spec)

    return err_spec


def residual_dwba(angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,
                  xi_l, xi_p, H, s, b, s_diff):
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    r = reflectivity.fields_debeyewaller(angle_in,angle_out,
                     wl,
                     stack,
                     henke,
                     period,
                     s_diff,
                     "henke_Si.csv")
    r1, t1, r2, t2, qx, qz, t, n = r
    spec = (r1, t1, r2, t2)
    dw = dwba.dwba_tilted_qx(spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b)
    if xi_l>0 and xi_p>0 and H>0 and H<=1.0 and s>0:
        err = y_meas/y_meas_err - omega*np.real(dw)/y_meas_err
    else:
        err = (y_meas/y_meas_err - omega*np.real(dw)/y_meas_err)*1e6
    return err, omega*np.real(dw), y_meas, y_meas_err

def residual_dwba_sk(angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,
                  xi_l, xi_p, H, s, b, s_diff):
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    r = reflectivity.fields_debeyewaller(angle_in,angle_out,
                     wl,
                     stack,
                     henke,
                     period,
                     s_diff,
                     "henke_Si.csv")
    r1, t1, r2, t2, qx, qz, t, n = r
    spec = (r1, t1, r2, t2)
    dw = dwba.dwba_tilted_qx_sk(spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b)
    if xi_l>0 and xi_p>0 and H>0 and H<=1.0 and s>0:
        err = y_meas/y_meas_err - omega*np.real(dw)/y_meas_err
    else:
        err = (y_meas/y_meas_err - omega*np.real(dw)/y_meas_err)*1e6
    return err, omega*np.real(dw), y_meas, y_meas_err

def eval_dwba(angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,
                  xi_l, xi_p, H, s, b, s_diff):
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    r = reflectivity.fields_debeyewaller(angle_in,angle_out,
                     wl,
                     stack,
                     henke,
                     period,
                     s_diff,
                     "henke_Si.csv")
    r1, t1, r2, t2, qx, qz, t, n = r
    spec = (r1, t1, r2, t2)
    dw = dwba.dwba_tilted_qx(spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b)
    return omega*np.real(dw)

def residual_dwba_densities(angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,densities,
                     capthicknesses,
                     caphenkes,
                     capdensities,
                     sigma,
                     xi_l, xi_p, H, s, b):
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    r = reflectivity.fields_dwba(angle_in,angle_out,
                     wl,
                     stack,
                     henke,
                     period,
                     None, sigma,
                     "Si",
                     densities,
                     None, 0,
                     capthicknesses,
                     caphenkes,
                     capdensities)
    r1, t1, r2, t2, qx, qz, t, n = r
    spec = (r1, t1, r2, t2)

    dw = dwba.dwba(spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b)
    err = y_meas/y_meas_err - omega*np.real(dw)/y_meas_err

    return err, omega*np.real(dw), y_meas, y_meas_err

def eval_dwba_densities(angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,densities,
                     capthicknesses,
                     caphenkes,
                     capdensities,
                     sigma,
                     xi_l, xi_p, H, s, b):
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    r = reflectivity.fields_debeyewaller_densities(angle_in,angle_out,
                     wl,
                     stack,
                     henke,
                     period,
                     sigma,
                     "henke_Si.csv",
                     densities,
                     capthicknesses,
                     caphenkes,
                     capdensities)
    r1, t1, r2, t2, qx, qz, t, n = r
    spec = (r1, t1, r2, t2)
    dw = dwba.dwba_tilted_qx(spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b)
    return omega*np.real(dw)


#def eval_dwba(p, arguments):

    #henke, period, layer_num_per_unit, aoi_spec, wl_spec, y_spec, diffuse = arguments
    
    #stack = p[:layer_num_per_unit]
    #s, xi_l, xi_p, H, b = p[layer_num_per_unit:]
    
    #r=250.0
    #wx=4.5
    #wy=4.5
    
    #omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    #res = []
    #pool = Pool(len(diffuse)+1 if (len(diffuse)+1)<17 else 16)
    
    #for cut in diffuse:
        #angle_in, angle_out, wl, y_meas, y_meas_err = cut
        #r = reflectivity.fields(angle_in,angle_out,
                        #wl,
                        #stack,
                        #henke,
                        #period,
                        #s,
                        #"henke_Si.csv")
        #r1, t1, r2, t2, qx, qz, t, n = r
        #spec = (r1, t1, r2, t2)
        #dw = pool.apply_async(dwba.dwba_tilted_qx, [spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b])
        #res.append(dw)
  
    #res_dwba = []
    #for worker in res:
        #res_dwba.append(worker.get())
        
    #pool.close()
    #pool.join()
    
    #return np.real(res_dwba)*omega


def residual(p, henke, period, layer_num_per_unit, aoi_spec, wl_spec, y_spec, diffuse):
    """
    @param p: list of fit parameters:
              (stack_thicknesses, henke_files, sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    pool = Pool(len(diffuse)+1 if (len(diffuse)+1)<17 else 16)
    
    stack = p[:layer_num_per_unit]
    s, xi_l, xi_p, H, b = p[layer_num_per_unit:]

    pool_res = []
    pool_res.append(pool.apply_async(residual_spec, [aoi_spec, wl_spec, y_spec, stack, henke, period, s]))

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba, [angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,xi_l, xi_p, H, s, b]))
   
  
    print("parameters:")
    print p
    results = []
    for res in pool_res:
        results.append(res.get())

    xi = np.concatenate(results)
    print xi.shape

    pool.close()
    pool.join()
    print np.sum(np.abs(xi)**2)/len(xi)
    return xi

def residual_log(p, henke, period, layer_num_per_unit, aoi_spec, wl_spec, y_spec, diffuse):
    """
    @param p: list of fit parameters:
              (stack_thicknesses, henke_files, sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    pool = Pool(len(diffuse)+1 if (len(diffuse)+1)<17 else 16)
    
    stack = p[:layer_num_per_unit]
    s, xi_l, xi_p, H, b = p[layer_num_per_unit:]

    pool_res = []
    pool_res.append(pool.apply_async(residual_spec_log, [aoi_spec, wl_spec, y_spec, stack, henke, period, s]))

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba, [angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,xi_l, xi_p, H, s, b]))
   
  
    print("parameters:")
    print p
    results = []
    for res in pool_res:
        results.append(res.get())

    xi = np.concatenate(results)
    print xi.shape

    pool.close()
    pool.join()
    print np.sum(np.abs(xi)**2)/len(xi)
    return xi

def residual_dwba_only(p, henke, stack,period,  s_diff, diffuse, fig):
    """
    @param p: list of fit parameters:
              (sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    cl = Client()
    lbv = cl.load_balanced_view()
    pool = lbv
    
    s, xi_l, xi_p, H, b = p

    pool_res = []

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba, angle_in,angle_out, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
    print p
    results = []
    dws = []
    for res in pool_res:
        err, dw = res.get()
        results.append(err)
        dws.append(dw)

    xi = np.concatenate(results)
    
    if not fig:
       fig = plt.figure()
    else:
       fig.clf()
       
    for i in xrange(len(dws)):
        angle_in, angle_out, wl, y_meas, y_meas_err = diffuse[i]
        ax = fig.add_subplot(4,2,i+1)
        if type(angle_in)!=float:
            ax.plot(angle_in, y_meas, 'ro')
            ax.plot(angle_in, dws[i], 'bo')
        else:
            ax.plot(wl, y_meas, 'ro')
            ax.plot(wl, dws[i], 'bo')
    fig.canvas.draw()
    
    print np.sum(np.abs(xi)**2)/len(xi)
    cl.close()
    return xi

def residual_dwba_only_mean3(p, henke, stack,period,  s_diff, diffuse):
    """
    @param p: list of fit parameters:
              (sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    cl = Client(profile='inner')
    lbv = cl.load_balanced_view()
    pool = lbv
    
    s, xi_l, xi_p, H, b = p

    pool_res_m05 = []
    pool_res_p05 = []
    pool_res = []

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba, angle_in,angle_out, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
            pool_res_m05.append(pool.apply_async(residual_dwba, angle_in,angle_out-0.5, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
            pool_res_p05.append(pool.apply_async(residual_dwba, angle_in,angle_out+0.5, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
    #print p
    results = []
    dws = []
    for i in xrange(len(pool_res)):
        err, dw, y_meas, y_meas_err = pool_res[i].get()
        errm05, dwm05, y_meas, y_meas_err = pool_res_m05[i].get()
        errp05, dwp05, y_meas, y_meas_err = pool_res_p05[i].get()
        avg = (dw+dwm05+dwp05)/3.0
        if xi_l>0 and xi_p>0 and H>0 and H<=1.0 and s>0:
            err_avg = y_meas/y_meas_err - avg/y_meas_err
        else:
            err_avg = (y_meas/y_meas_err - avg/y_meas_err)*1e6
        results.append(err_avg)
        dws.append(avg)

    xi = np.concatenate(results)
    
    #if not fig:
       #fig = plt.figure()
    #else:
       #fig.clf()
       
    #for i in xrange(len(dws)):
        #angle_in, angle_out, wl, y_meas, y_meas_err = diffuse[i]
        #ax = fig.add_subplot(4,2,i+1)
        #if type(angle_in)!=float:
            #ax.loglog(angle_in, y_meas, 'ro')
            #args = np.argsort(angle_in)
            #ax.plot(angle_in[args], dws[i][args], 'b-', lw=2)
        #else:
            #ax.plot(wl, y_meas, 'ro')
            #args = np.argsort(wl)
            #ax.plot(wl[args], dws[i][args], 'b-', lw=2)
    #fig.canvas.draw()
    
    #print np.sum(np.abs(xi)**2)/len(xi)
    cl.close()
    return xi

def residual_dwba_only_mean3_densities(p, stack, henke, period, densities,  capthicknesses,
                                                                            caphenkes,
                                                                            capdensities,
                                                                            sigma, diffuse):
    """
    @param p: list of fit parameters:
              (sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    cl = Client(profile='inner')
    lbv = cl.load_balanced_view()
    pool = lbv
    
    s, xi_l, xi_p, H, b = p

    pool_res_m05 = []
    pool_res_p05 = []
    pool_res = []

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba_densities, angle_in,angle_out, wl, y_meas, y_meas_err, 
                                                                            stack, henke, period, densities,
                                                                            capthicknesses,
                                                                            caphenkes,
                                                                            capdensities,
                                                                            sigma,
                                                                            xi_l, xi_p, H, s, b))
            pool_res_m05.append(pool.apply_async(residual_dwba_densities, angle_in,angle_out-0.5, wl, y_meas, y_meas_err, 
                                                                            stack, henke, period, densities,
                                                                            capthicknesses,
                                                                            caphenkes,
                                                                            capdensities,
                                                                            sigma,
                                                                            xi_l, xi_p, H, s, b))
            pool_res_p05.append(pool.apply_async(residual_dwba_densities, angle_in,angle_out+0.5, wl, y_meas, y_meas_err, 
                                                                            stack, henke, period, densities,
                                                                            capthicknesses,
                                                                            caphenkes,
                                                                            capdensities,
                                                                            sigma,
                                                                            xi_l, xi_p, H, s, b))
    #print p
    results = []
    dws = []
    for i in xrange(len(pool_res)):
        err, dw, y_meas, y_meas_err = pool_res[i].get()
        errm05, dwm05, y_meas, y_meas_err = pool_res_m05[i].get()
        errp05, dwp05, y_meas, y_meas_err = pool_res_p05[i].get()
        avg = (dw+dwm05+dwp05)/3.0
        if xi_l>0 and xi_p>0 and H>0 and H<=1.0 and s>0:
            err_avg = y_meas/y_meas_err - avg/y_meas_err
        else:
            err_avg = (y_meas/y_meas_err - avg/y_meas_err)*1e6
        results.append(err_avg)
        dws.append(avg)

    xi = np.concatenate(results)
    
    #if not fig:
       #fig = plt.figure()
    #else:
       #fig.clf()
       
    #for i in xrange(len(dws)):
        #angle_in, angle_out, wl, y_meas, y_meas_err = diffuse[i]
        #ax = fig.add_subplot(4,2,i+1)
        #if type(angle_in)!=float:
            #ax.loglog(angle_in, y_meas, 'ro')
            #args = np.argsort(angle_in)
            #ax.plot(angle_in[args], dws[i][args], 'b-', lw=2)
        #else:
            #ax.plot(wl, y_meas, 'ro')
            #args = np.argsort(wl)
            #ax.plot(wl[args], dws[i][args], 'b-', lw=2)
    #fig.canvas.draw()
    
    #print np.sum(np.abs(xi)**2)/len(xi)
    cl.close()
    return xi

def residual_dwba_only_mean3_plot(p, henke, stack,period,  s_diff, diffuse, fig):
    """
    @param p: list of fit parameters:
              (sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    cl = Client(profile='inner')
    lbv = cl.load_balanced_view()
    pool = lbv
    
    s, xi_l, xi_p, H, b = p

    pool_res_m05 = []
    pool_res_p05 = []
    pool_res = []

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba, angle_in,angle_out, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
            pool_res_m05.append(pool.apply_async(residual_dwba, angle_in,angle_out-0.5, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
            pool_res_p05.append(pool.apply_async(residual_dwba, angle_in,angle_out+0.5, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
    #print p
    results = []
    dws = []
    for i in xrange(len(pool_res)):
        err, dw, y_meas, y_meas_err = pool_res[i].get()
        errm05, dwm05, y_meas, y_meas_err = pool_res_m05[i].get()
        errp05, dwp05, y_meas, y_meas_err = pool_res_p05[i].get()
        avg = (dw+dwm05+dwp05)/3.0
        if xi_l>0 and xi_p>0 and H>0 and H<=1.0 and s>0:
            err_avg = y_meas/y_meas_err - avg/y_meas_err
        else:
            err_avg = (y_meas/y_meas_err - avg/y_meas_err)*1e6
        results.append(err_avg)
        dws.append(avg)

    xi = np.concatenate(results)
    
    if not fig:
       fig = plt.figure()
    else:
       fig.clf()
       
    for i in xrange(len(dws)):
        angle_in, angle_out, wl, y_meas, y_meas_err = diffuse[i]
        ax = fig.add_subplot(4,2,i+1)
        if type(angle_in)!=float:
            ax.loglog(angle_in, y_meas, 'ro')
            args = np.argsort(angle_in)
            ax.plot(angle_in[args], dws[i][args], 'b-', lw=2)
        else:
            ax.plot(wl, y_meas, 'ro')
            args = np.argsort(wl)
            ax.plot(wl[args], dws[i][args], 'b-', lw=2)
    fig.canvas.draw()
    
    print np.sum(np.abs(xi)**2)/len(xi)
    cl.close()
    return xi

def residual_dwba_only_mean3_sk(p, henke, stack,period,  s_diff, diffuse, fig):
    """
    @param p: list of fit parameters:
              (sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    cl = Client()
    lbv = cl.load_balanced_view()
    pool = lbv
    
    s, xi_l, xi_p, H, b = p

    pool_res_m05 = []
    pool_res_p05 = []
    pool_res = []

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(residual_dwba_sk, angle_in,angle_out, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
            pool_res_m05.append(pool.apply_async(residual_dwba_sk, angle_in,angle_out-0.5, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
            pool_res_p05.append(pool.apply_async(residual_dwba_sk, angle_in,angle_out+0.5, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
    print p
    results = []
    dws = []
    for i in xrange(len(pool_res)):
        err, dw, y_meas, y_meas_err = pool_res[i].get()
        errm05, dwm05, y_meas, y_meas_err = pool_res_m05[i].get()
        errp05, dwp05, y_meas, y_meas_err = pool_res_p05[i].get()
        avg = (dw+dwm05+dwp05)/3.0
        if xi_l>0 and xi_p>0 and H>0 and H<=1.0 and s>0:
            err_avg = y_meas/y_meas_err - avg/y_meas_err
        else:
            err_avg = (y_meas/y_meas_err - avg/y_meas_err)*1e6
        results.append(err_avg)
        dws.append(avg)

    xi = np.concatenate(results)
    
    if not fig:
       fig = plt.figure()
    else:
       fig.clf()
       
    for i in xrange(len(dws)):
        angle_in, angle_out, wl, y_meas, y_meas_err = diffuse[i]
        ax = fig.add_subplot(4,2,i+1)
        if type(angle_in)!=float:
            ax.plot(angle_in, y_meas, 'ro')
            args = np.argsort(angle_in)
            ax.plot(angle_in[args], dws[i][args], 'b-', lw=2)
        else:
            ax.plot(wl, y_meas, 'ro')
            args = np.argsort(wl)
            ax.plot(wl[args], dws[i][args], 'b-', lw=2)
    fig.canvas.draw()
    
    print np.sum(np.abs(xi)**2)/len(xi)
    cl.close()
    return xi

def eval_dwba_only(p, shift, henke, stack,period,  s_diff, diffuse, fig):
    """
    @param p: list of fit parameters:
              (sigma, xi_l, xi_p, H, beta)
    
    @param diffuse: diffuse is a list of equally long np arrays of the shape
                    [[angle_in, angle_out, wavelength, y_meas, y_meas_err], ...]
    """
    cl = Client()
    lbv = cl.load_balanced_view()
    pool = lbv
    
    s, xi_l, xi_p, H, b = p

    pool_res = []

    if len(diffuse) == 0:
        pass
    else:
        for cut in diffuse:
            angle_in, angle_out, wl, y_meas, y_meas_err = cut
            pool_res.append(pool.apply_async(eval_dwba, angle_in,angle_out+shift, wl, y_meas, y_meas_err,
                                                                          stack, henke, period,xi_l, xi_p, H, s, b, s_diff))
   
  
    results = []
    for res in pool_res:
        results.append(res.get())
        
    if not fig:
       fig = plt.figure()
    else:
       fig.clf()
       
    for i in xrange(len(results)):
        angle_in, angle_out, wl, y_meas, y_meas_err = diffuse[i]
        ax = fig.add_subplot(4,2,i+1)
        if type(angle_in)!=float:
            ax.plot(angle_in, y_meas, 'ro')
            ax.plot(angle_in, results[i], 'bo')
        else:
            ax.plot(wl, y_meas, 'ro')
            ax.plot(wl, results[i], 'bo')
    fig.canvas.draw()
    cl.close()
    return results


def deriv_dwba_only(p, henke, stack,period,  s_diff, diffuse):
    derivparams = []
    delta = 1e-2
    
    cl = Client()
    lbv = cl.load_balanced_view()
    pool = lbv
    
    zero_handle = pool.apply_async(eval_dwba_only, p, henke, stack,period,  s_diff, diffuse)
    
    for i in range(len(p)):
        copy = np.array(p)
        copy[i] += delta
        derivparams.append(copy)
        
    result = []
    handles = []
    
    for dp in derivparams:
        handles.append(pool.apply_async(eval_dwba_only, dp, henke, stack,period,  s_diff, diffuse))
    
    for handle in handles:
        result.append(handle.get())
    
    zero = zero_handle.get()
    
    return [(np.concatenate(r) - np.concatenate(zero))/delta for r in result]
    


def fit(parameters, arguments, stepwidth, tolerance):
    return leastsq(residual, parameters, args=arguments, full_output=True, epsfcn=stepwidth, xtol=tolerance)

def fit_log(parameters, arguments, stepwidth, tolerance):
    return leastsq(residual_log, parameters, args=arguments, full_output=True, epsfcn=stepwidth, xtol=tolerance)

def fit_spec(parameters, arguments, stepwidth, tolerance):
    return leastsq(residual_spec, parameters, args=arguments, full_output=True, epsfcn=stepwidth, xtol=tolerance)


def fit_dwba_only(parameters, arguments, tolerance):
    return leastsq(residual_dwba_only, parameters, args=arguments, full_output=True, ftol=tolerance)
    
    