import reflectivity as refl
import numpy as np
import helper

en = np.linspace(500,500,1)
#r, t = xrr([0.0],
#    nm2eV(en),
#    [5,460,5],
#    ["henke_CrO2_500-770.csv","henke_Cr_500-770.csv","henke_CrO2_500-770.csv"],
#    1,
#    0.0,
#    "henke_vac.csv",
#    [0.9,0.9,0.9],
#    None, None, None)
r2, t2, n, layer = refl.fields([0.0],
    helper.nm2eV(en),
    [5,410,5],
    ["CrO2","Cr","CrO2"],
    1, [1.0,1.0,1.0,1.0],
    0.0,
    "vac",
    [1.0,1.0,1.0], [0.25,0.25,0.25,0.25], 2,
    None, None, None)

pass