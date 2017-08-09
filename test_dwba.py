import numpy as np
import reflectivity
import dwba

def eval_dwba_qx_mixlayer_fast(p, thicknesses, henke, roughness, wavelength, angle_in, angle_out, sigma, densities, intermix,capthicknesses, caphenkes, capdensities):
    s, xi_l , xi_p,H, beta = p
    r=250.0
    wx=4.5
    wy=4.5
    
    omega = 4*np.arctan(wx*wy/(2*r*np.sqrt(4*r**2+wx**2+wy**2)))
    #omega=1
    r = reflectivity.fields_dwba(angle_in,angle_out,
                     wavelength,
                     thicknesses,
                     henke,
                     200,
                     sigma, roughness,
                     "Si", densities, intermix, 3,capthicknesses, caphenkes,capdensities)
    r1, t1, r2, t2, qx, qz, t, n = r
    print r1.shape
    spec = (r1, t1, r2, t2)
    dw = dwba.dwba_tilted_qx_fast(spec, qx, t, n, wavelength, qz, xi_l, xi_p, angle_in, H, s, beta)
    return omega*np.real(dw)

def replicate(array, number):
    if type(array)!=type([]):
        return np.outer(np.ones(number),array)
    else:
        return number*array
    
capoxide, cap, sc, cr, roughness = [0.95823542, 2.8455255, 0.9018008237443974, 0.6711297662556025, 0.0]
p_res = [ 1.57293059,  0.57332525,  0.30007203,  1.02264646,  0.07860989]
#p_res = [ 1.5741015 ,  0.64572049,  0.26592892,  1.0153262 ,  1.04042057]
D, Gamma_sc, mix, sigma_diffusion, sigma_gamma = p_res
densities = [[1.0,1.0]+[1.0,1.0]*400]
intermix = [[0.0,0.0]+[mix,mix]*400+[mix]]
sigma = np.array([0.2,0.5]+[sigma_diffusion*sigma_gamma,sigma_diffusion*(1-sigma_gamma)]*400+[0.4])
sigma = [sigma.reshape((len(sigma),1))]
roughness = 0.0

t = [[sc,cr]]
h = [["Sc","Cr"]]
ct = [[capoxide, cap]]
ch =[["CrO2","Cr"]]
cd = [[1.0,1.0]]
ls = np.linspace(3.1,3.16,100)
angle_in = np.linspace(-4,7,200)
angle_out = 3-angle_in
p=[0.22105716398682855, 4.4497017853644527, 9.8500669187536705, 0.59354184407475219, 0.49577981703771951]

W = replicate(ls,len(angle_out))
P = replicate(np.array(p),len(angle_out))
T = replicate(t,len(angle_out))
H = replicate(h,len(angle_out))
S = replicate(sigma,len(angle_out))
R = replicate(roughness,len(angle_out))
D = replicate(densities,len(angle_out))
I = replicate(intermix,len(angle_out))
CT = replicate(ct,len(angle_out))
CH = replicate(ch,len(angle_out))
CD = replicate(cd,len(angle_out))

res1 = eval_dwba_qx_mixlayer_fast(P[0],T[0],H[0],R[0],W[0],angle_in[0],angle_out[0],S[0],D[0],I[0],CT[0],CH[0],CD[0])