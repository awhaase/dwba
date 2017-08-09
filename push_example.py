from ipyparallel import Client


# Set up the interface to the ipcluster.
c = Client(profile='main', retries=5)
view = c[:]
lbview = c.load_balanced_view()
def push():
    view.execute("import numpy as np")
    view.execute("import reflectivity")
    view.execute("import helper")
    view.push({"lnlike": lnlike})
    view.push({"lnprior": lnprior})
    view.push({"objective": objective})
    view.push({"cols" : cols})

    view.push({"lnprob": lnprob, 
               "residual_mosi" : residual_mosi, 
               "residual_mosi_fixMo": residual_mosi_fixMo,
               "residual_dwba" : residual_dwba,
               "eval_dwba" : eval_dwba,
               "lb" : lb, 
               "ub" : ub, 
               "arguments": arguments,
               "cuts" : cuts, "structure" : structure})
    print len(view)
push()