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
    
    def run(self, arguments,maxiter=100, fig=None, savename='pso', verbose=True, hawk=100, plotmodulo=10, silent=False, plot_save=True):

        
        cl = Client(profile='main')
        pool = cl.load_balanced_view()
        pool.set_flags(retries=3)
        
        if verbose:
            print "Master pool initialized: %i  cores" % len(pool)
        
        #cl2 = Client(profile='inner')
        #pool2 = cl2.load_balanced_view()
        #if verbose:
            #print "Slave pool initialized: %i cores" % len(pool2)
        
        iter = 0
        
        if plot_save:
            ax1 = fig.add_subplot(341)
            ax2 = fig.add_subplot(342)
            ax3 = fig.add_subplot(343)
            ax4 = fig.add_subplot(344)
            ax5 = fig.add_subplot(345)
            ax6 = fig.add_subplot(346)
            ax7 = fig.add_subplot(347)
            ax8 = fig.add_subplot(348)
            ax9 = fig.add_subplot(349)
            ax10 = fig.add_subplot(3,4,10)
            ax11 = fig.add_subplot(3,4,11)
        
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
            # print "Particle %i" % i
                #print self.X[i]
                #args = (self.X[i], arguments[0],arguments[1],arguments[2],arguments[3],arguments[4])
                if arguments:
                    args = (self.X[i],) + arguments
                    handles.append(pool.apply_async(self.obj_func, *args))
                else:
                    handles.append(pool.apply_async(self.obj_func, self.X[i]))
                
                #print "submitted"
            res = []
            
            new_xi = []
           
            for i in xrange(len(handles)):
                hdlres = handles[i].get()
                tmp, euv, xrr =hdlres, None, None
                new_xi = 10*(np.sum(np.abs(euv)**2)/len(euv)) + (np.sum(np.abs(xrr)**2)/len(xrr))
                if ((new_xi < self.xi[i]) | (self.xi[i] <= 0.0)):
                    self.P_id[i] = self.X[i]
                    #ich glaube hier war ein BUG
                    self.xi[i] = new_xi
                    #BEGIN DEBUG
                    self.xi_euv[i] = 10*(np.sum(np.abs(euv)**2)/len(euv))
                    self.xi_xrr[i] = (np.sum(np.abs(xrr)**2)/len(xrr))
                    #END DEBUG
                if verbose:
                    print "Particle %i: xi=%.2f" % (i, new_xi)
                #save particle xi and parameters
                
                
                
                
            best_global = np.argmin(self.xi)
            
            if ((np.min(self.xi)<self.xi_best_gloabl) | (self.xi_best_gloabl<=0.0)):
                self.P_gl = self.X[best_global]
                self.xi_best_gloabl = self.xi[best_global]
                call_the_hawk = 0
                if not silent:
                    print "%i: new global minimum found xi = %f, euv= %f, xrr= %f, hawk put back to sleep" % (iter, self.xi_best_gloabl, self.xi_euv[best_global],  self.xi_xrr[best_global])
                    print self.P_gl
            
            self._calculateVelocity()
            if verbose:
                print "Best global xi: %f" % self.xi_best_gloabl
            
                print "Best global particle:"
                print self.P_gl
            if plot_save:
                if iter%plotmodulo==0:
                    pos_mo=[]
                    pos_si=[]
                    pos_b4c=[]
                    pos_c=[]
                    pos_sigma = []
                    for i in xrange(self.N):
                        pos_si.append((np.array(self.X[i]))[0])
                        pos_b4c.append((np.array(self.X[i]))[1])
                        pos_mo.append((np.array(self.X[i]))[2])
                        pos_c.append((np.array(self.X[i]))[3])
                        pos_sigma.append((np.array(self.X[i]))[4])


                    ax1.plot(iter, np.sum(self.xi)/self.N, 'ro-')
                    ax1.set_ylabel("average $\\chi$")
                    ax1.set_xlabel("iteration")
                    
                    ax2.plot(iter, np.sum(np.abs(self.V_id)), 'bo-')
                    ax2.set_ylabel("total velocity")
                    ax2.set_xlabel("iteration")
                    
                    ax3.plot(iter, self.xi_best_gloabl, 'go-')
                    ax3.set_ylabel("best global $\\chi$")
                    ax3.set_xlabel("iteration")
                    
                    pos = pos_si
                    axes = ax4
                    for ps in pos:
                        axes.plot(iter, ps, 'k,', markersize=2.5)
                    #axes.plot(iter, np.mean(pos), 'ko', markersize=2.5)
                    axes.plot(iter, min(pos), 'k^', markersize=2.5)
                    axes.plot(iter, max(pos), 'kv', markersize=2.5)
                    axes.plot(iter, self.P_gl[0], 'ro', markersize=3.0, mec='r')
                    axes.set_ylabel("pos si")
                    axes.set_xlabel("iteration")
                    
                    pos = pos_mo
                    axes = ax5
                    for ps in pos:
                        axes.plot(iter, ps, 'k,', markersize=2.5)
                    #axes.plot(iter, np.mean(pos), 'ko', markersize=2.5)
                    axes.plot(iter, min(pos), 'k^', markersize=2.5)
                    axes.plot(iter, max(pos), 'kv', markersize=2.5)
                    axes.plot(iter, self.P_gl[2], 'ro', markersize=3.0, mec='r')
                    axes.set_ylabel("pos mo")
                    axes.set_xlabel("iteration")
                    
                    pos = pos_b4c
                    axes = ax6
                    for ps in pos:
                        axes.plot(iter, ps, 'k,', markersize=2.5)
                    #axes.plot(iter, np.mean(pos), 'ko', markersize=2.5)
                    axes.plot(iter, min(pos), 'k^', markersize=2.5)
                    axes.plot(iter, max(pos), 'kv', markersize=2.5)
                    axes.plot(iter, self.P_gl[1], 'ro', markersize=3.0, mec='r')
                    axes.set_ylabel("pos b4c")
                    axes.set_xlabel("iteration")
                    
                    pos = pos_c
                    axes = ax7
                    for ps in pos:
                        axes.plot(iter, ps, 'k,', markersize=2.5)
                    #axes.plot(iter, np.mean(pos), 'ko', markersize=2.5)
                    axes.plot(iter, min(pos), 'k^', markersize=2.5)
                    axes.plot(iter, max(pos), 'kv', markersize=2.5)
                    axes.plot(iter, self.P_gl[3], 'ro', markersize=3.0, mec='r')
                    axes.set_ylabel("pos c")
                    axes.set_xlabel("iteration")
                    
                    pos = pos_sigma
                    axes = ax8
                    for ps in pos:
                        axes.plot(iter, ps, 'k,', markersize=2.5)
                    #axes.plot(iter, np.mean(pos), 'ko', markersize=2.5)
                    axes.plot(iter, min(pos), 'k^', markersize=2.5)
                    axes.plot(iter, max(pos), 'kv', markersize=2.5)
                    axes.plot(iter, self.P_gl[4], 'ro', markersize=3.0, mec='r')
                    axes.set_ylabel("pos sigma")
                    axes.set_xlabel("iteration")
                    
                    euv, xrr, xrr2 = self.eval_func(self.P_gl, *arguments)
                    
                    ax9.cla()
                    ax9.plot(arguments[0],arguments[1], 'r-')
                    ax9.plot(arguments[0],euv, 'b-')
                    
                    ax10.cla()
                    ax10.plot(arguments[2],arguments[3], 'r-')
                    ax10.plot(arguments[2],xrr, 'b-')
                    ax10.set_yscale('log')
                    
                    ax11.cla()
                    ax11.plot(arguments[4],arguments[5], 'r-')
                    ax11.plot(arguments[4],xrr2, 'b-')
                    ax11.set_yscale('log')

                    
                    ax1.set_yscale('log')
                    ax2.set_yscale('log')
                    ax3.set_yscale('log')
                    #ax4.set_yscale('linear')
                    fig.canvas.draw()
                    fig.savefig("/home/ahaase/Dropbox/%s.pdf" % savename, format='PDF')
            
        cl.purge_everything()
        pool.results.clear()
        cl.results.clear()
        cl.metadata.clear()
        cl.close()
        del cl
        del pool
        print "Best global xi: %f" % self.xi_best_gloabl
        print self.P_gl
        return self.P_gl
    
    def run_dwba(self, arguments,maxiter=100, fig=None, savename='pso', verbose=True, hawk=100, plotmodulo=10, silent=False, plot_save=True, external_check=None):

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
    
    