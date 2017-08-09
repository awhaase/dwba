import reflectivity
import numpy as np
from scipy.constants import c,pi
from scipy.constants import physical_constants
h_planck = physical_constants['Planck constant in eV s'][0]
def nm2eV(wl):
    return h_planck*c/(wl*1E-9)


energies = np.array([ 396.39704702,  397.39720259,  398.39735816,  399.39751373,
        400.3976693 ,  401.39782487,  402.39798044,  403.39813601,
        404.39829158,  405.39844715,  406.39860272,  407.39875829,
        408.39891386,  409.39906943,  410.399225  ,  411.39938057,
        412.39953614,  413.39969171,  414.39984728])
wls = nm2eV(energies)
an = [np.array([ 14.03736816,  14.13669254,  14.23698201,  14.33670852,
         14.43780224,  14.43675672,  14.53833299,  14.63673249,
         14.73750451,  14.83811569,  14.93752049,  15.13572692,
         15.13685286,  15.23710213,  15.33682863,  15.4367562 ,
         15.43699747,  15.5368044 ,  15.63689281,  15.73706165,
         15.83690879,  15.93679615,  16.03672371,  16.1370534 ,
         16.23661906,  16.33727044,  16.43615249,  16.53704515,
         16.63657059,  16.73702092,  16.83658657,  16.93687605,
         17.03700468,  17.13701267,  17.23698044,  17.33710907,
         17.43703664,  17.53676314,  17.63669071,  17.73766379])]





param = [2.0,1.0,0.5,1.0,1.0,1.0,0.0,0.0,0.0]
D, sc,b4c, mixsc, mixb4c, mixcr, sigma_sc, sigma_b4c, sigma_cr = param
capoxide, cap = 0.95823542,  2.8455255
cr, sigma_rough = [ D-sc-b4c,  0.0]

densities = [1.0,1.0]+[1.0,1.0,1.0]*300
intermix = [0.0,mixcr]+[mixsc,mixb4c,mixcr]*300+[mixsc]
sigma = np.array([0.1,0.1]+[sigma_sc,sigma_b4c,sigma_cr]*300+[0.1])
sigma = sigma.reshape((len(sigma),1))
stack = [sc,b4c,cr]
data = []
ener = []
angl = []
ydata_c = None
# if any([p<0 for p in param]+[param[3]>1.0,param[4]>1.0,param[5]>1.0]):
#     pass
# for i in xrange(len(wls)):
#     r = reflectivity.xrr(an[i],
#                  np.ones(len(an[i]))*np.array([wls[i]]),
#                  stack,
#                  ["Sc","B4C","Cr"],
#                  300,
#                  sigma, 0.0,
#                  "Si",densities, intermix ,15,[capoxide, cap], ["CrO2","Cr"],[1.0,1.0])
#     data.append(r)
#     ener.append(nm2eV(np.ones(len(an[i]))*np.array([wls[i]])))
#     angl.append(an[i])
# fitdata_c = np.concatenate(data)
# xis = np.array(np.log(ydata_c) - np.log(fitdata_c))
# print np.sum(xis**2)/len(xis)

a=np.array([0.0,0.0,0.0]+[1.0,0.0,0.0]*300+[0.0])
th2a=np.array([0.0,0.95,2.84]+[sc,b4c,cr]*300+[0.0])
th2a = th2a.reshape(len(th2a),1)

a,ths = reflectivity.generate_mixlayers(a, th2a, intermix, sigma,15)
a = a.reshape(len(a),1)

pass