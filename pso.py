import numpy as np
import random
import matplotlib as plt
import csv
from ipyparallel import Client

class Schwarm:
    
    
    
    def __init__(self, obj_func, N, K, eval_func=None):
        self.obj_func = obj_func
        self.N = N
        self.K = K
        self.xi = [-1.0 for i in xrange(N)]
        
        #DEBUG
        #self.xi_euv = [-1.0 for i in xrange(N)]
        #self.xi_xrr = [-1.0 for i in xrange(N)]
        
        self.X  = [[] for i in xrange(N)]
        self.P_id =  [[] for i in xrange(N)]
        self.P_gl = []
        self.V_id = [[] for i in xrange(N)]
        self.xi_best_gloabl = -1.0
        self.eval_func = eval_func
        self._externalCheckFunction = None
        
    def _checkBoundaries(self, i):
        if self._externalCheckFunction is not None:
            external = self._externalCheckFunction(np.array(self.X[i])+np.array(self.V_id[i]))
        else:
            external = True      
        return all( ((np.array(self.X[i])+np.array(self.V_id[i]))>= np.array(self.lower_boundary)) & ((np.array(self.X[i])+np.array(self.V_id[i]))<= np.array(self.upper_boundary)) & external)
        
    def initialize(self, lower_boundary_vector, upper_boundary_vector, weights):
        self.lower_boundary = lower_boundary_vector
        self.upper_boundary = upper_boundary_vector
        self.weights = weights
        self.xi = [-1.0 for i in xrange(self.N)]
        
        for i in xrange(self.N):
            self.X[i] = [random.uniform(lower_boundary_vector[j],upper_boundary_vector[j]) for j in xrange(len(lower_boundary_vector))]
            
            while True:
                self.V_id[i] = [random.uniform(-1.0,1.0)*weight for weight in self.weights]
                if self._checkBoundaries(i):
                    break
                
    
    def reset_velocity(self, i):
        #self.V_id[i] = [-elem for elem in self.V_id[i]]
        ##print "reflected"
        #if not self._checkBoundaries(i):
        while True:
                self.V_id[i] = [random.uniform(-1.0,1.0)*weight for weight in self.weights]
                if self._checkBoundaries(i):
                    break
    
    #def set_limits(self, lower_boundary, upper_boundary):
    #    return
  
    def _calculateVelocity(self):
        phi_1 = 2.05
        phi_2 = 2.05
        phi = phi_1+phi_2
        #K = 2.0/np.abs(2- phi - np.sqrt(phi**2 - 4*phi))
        K = self.K
        #print "K: %f" % K
        for i in xrange(self.N):
            self.V_id[i] = list( K* (np.array(self.V_id[i]) 
                                     + phi_1*random.random()*(np.array(self.P_id[i]) - np.array(self.X[i])) 
                                     + phi_2*random.random()*(np.array(self.P_gl) - np.array(self.X[i])) ) )
            if not self._checkBoundaries(i):
                self.reset_velocity(i)
    def total_random(self, arguments,maxiter=100, savename='pso'):
        iter = 0
        cl = Client(profile='main')
        pool = cl.load_balanced_view()
        while iter<maxiter:
            iter+=1
            handles = []
            for i in xrange(self.N):
                self.X[i] = [random.uniform(self.lower_boundary[j],self.upper_boundary[j]) for j in xrange(len(self.lower_boundary))]  
                args = (self.X[i],) + arguments
                handles.append(pool.apply_async(self.obj_func, *args))
                
            with open("/home/ahaase/Dropbox/xilist_%s.csv" % savename, 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in xrange(len(handles)):
                        tmp = handles[i].get()
                        new_xi = (np.sum(np.abs(tmp)**2)/len(tmp))
                        spamwriter.writerow([new_xi] + self.X[i])
        cl.close()
    

    def run(self, arguments, maxiter=100, fig=None, savename='pso', verbose=True, hawk=100, plotmodulo=10, silent=False, plot_save=True, external_check=None):

        self._externalCheckFunction = external_check
        
        cl = Client(profile='main')
        pool = cl.load_balanced_view()
        
        if verbose:
            print "Master pool initialized: %i  cores" % len(pool)

        
        iter = 0
        
        
        import csv
        
        call_the_hawk = 0
                   
        while iter<maxiter:
            call_the_hawk += 1 
            diff = 0
            iter+=1
            if verbose:
                print "Begin iteration %i" % iter
            handles = []
            if call_the_hawk >= hawk:
                self.initialize(self.lower_boundary, self.upper_boundary, self.weights)
                if not silent:
                    print "%i: hawk woke up and killed all the birds, newborns underway" %iter
                call_the_hawk = 0
            for i in xrange(self.N):
                self.X[i] = [self.X[i][j] + self.V_id[i][j] for j in xrange(len(self.X[i]))]
                if arguments:
                    args = (self.X[i],) + arguments
                    handles.append(pool.apply_async(self.obj_func, *args))
                else:
                    handles.append(pool.apply_async(self.obj_func, self.X[i]))
                
            res = []
            
            new_xi = []
           
            for i in xrange(len(handles)):
                tmp = handles[i].get()
                new_xi = np.sum(np.abs(tmp))
                if ((new_xi < self.xi[i]) | (self.xi[i] <= 0.0)):
                    self.P_id[i] = self.X[i]
                    self.xi[i] = new_xi
                if verbose:
                    print "Particle %i: xi=%.2f" % (i, new_xi)

                
                
            best_global = np.argmin(self.xi)
            
            if ((np.min(self.xi)<self.xi_best_gloabl) | (self.xi_best_gloabl<=0.0)):
                self.P_gl = self.X[best_global]
                self.xi_best_gloabl = self.xi[best_global]
                call_the_hawk = 0
                if not silent:
                    print "%i: new global minimum xi = %e" % (iter, self.xi_best_gloabl)
                    print self.P_gl
            
            self._calculateVelocity()
            if verbose:
                print "Best global xi: %f" % self.xi_best_gloabl
            
                print "Best global particle:"
                print self.P_gl
            
            
        pool.results.clear()
        cl.results.clear()
        cl.metadata.clear()
        cl.close()
        print "Best global xi: %e" % self.xi_best_gloabl
        print self.P_gl
        return self.P_gl
    
    