__author__ = 'ahaase'
import reflectivity
reload(reflectivity)
import numpy as np
ang = np.linspace(0.01,9.0,700)
CU_KALPHA1 = 0.15405980
D, co, c, mix,Gamma_sc, sigma_diffusion, sigma_gamma, rough = [1.5713900702959465, 1.8, 1.8, 0.44387926906480497, 0.48291965634129608, 0.74283063898590473, 1.0, 0.25]
sc, cr = [D*Gamma_sc, D*(1-Gamma_sc)]
densities = [1.0,1.0]
intermix = [0.0,0.0]+[mix,mix]*400+[mix]
sigma = np.array([0.0,0.0]+[sigma_diffusion*sigma_gamma,sigma_diffusion*(1-sigma_gamma)]*400+[0.4])
sigma = sigma.reshape((len(sigma),1))
r = reflectivity.xrr(90-ang,
                 np.ones(len(ang))*CU_KALPHA1,
                 [sc,cr],
                 ["Sc","Cr"],
                 400,
                 sigma, rough,
                 "Si", densities,intermix, 15,[co, c],["CrO2","Cr"],[1.0,1.0])
