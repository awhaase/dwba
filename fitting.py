from scipy.optimize import leastsq, curve_fit
import numpy as np
import reflectivity
import dwba
from helper import *
from multiprocessing import Pool
import time


def residual_spec(angle_in, wl_spec, y_spec, stack, henke, period, s):
    r = reflectivity.xrr(angle_in,
                     wl_spec,
                     stack,
                     henke,
                     period,
                     s,
                     "henke_Si.csv")
     
    if all(thickness>0 for thickness in stack) and s>0:
        err_spec = np.log(y_spec/(0.01*np.nanmax(y_spec)))-np.log(r/(0.01*np.nanmax(y_spec)))
    else:
        err_spec = (np.log(y_spec/np.nanmax(y_spec))-np.log(r/np.nanmax(y_spec)))*1e6
    return err_spec


def residual_dwba(angle_in, angle_out, wl, y_meas, y_meas_err, stack, henke, period,
                  xi_l, xi_p, H, s, b):
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    
    r = reflectivity.fields(angle_in,angle_out,
                     wl,
                     stack,
                     henke,
                     period,
                     s,
                     "henke_Si.csv")
    r1, t1, r2, t2, qx, qz, t, n = r
    spec = (r1, t1, r2, t2)
    dw = dwba.dwba_tilted_qx(spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b)
    if xi_l>0 and 601>xi_p>0 and H>0 and H<=1.0 and s>0:
        err = y_meas/y_meas_err - omega*np.real(dw)/y_meas_err
    else:
        err = (y_meas/y_meas_err - omega*np.real(dw)/y_meas_err)*1e6
    return err

def eval_dwba(p, arguments):

    henke, period, layer_num_per_unit, aoi_spec, wl_spec, y_spec, diffuse = arguments
    
    stack = p[:layer_num_per_unit]
    s, xi_l, xi_p, H, b = p[layer_num_per_unit:]
    
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    res = []
    pool = Pool(len(diffuse)+1 if (len(diffuse)+1)<17 else 16)
    
    for cut in diffuse:
        angle_in, angle_out, wl, y_meas, y_meas_err = cut
        r = reflectivity.fields(angle_in,angle_out,
                        wl,
                        stack,
                        henke,
                        period,
                        s,
                        "henke_Si.csv")
        r1, t1, r2, t2, qx, qz, t, n = r
        spec = (r1, t1, r2, t2)
        dw = pool.apply_async(dwba.dwba_tilted_qx, [spec, qx, t, n, wl, qz, xi_l, xi_p, angle_in, H, s, b])
        res.append(dw)
  
    res_dwba = []
    for worker in res:
        res_dwba.append(worker.get())
        
    pool.close()
    pool.join()
    
    return np.real(res_dwba)*omega


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

def fit(parameters, arguments, stepwidth, tolerance):
    return leastsq(residual, parameters, args=arguments, full_output=True, epsfcn=stepwidth, xtol=tolerance)