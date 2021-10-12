from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import astra

# Imports from original SART.py
import argparse
from glob import glob
import astra
import numpy as np
import os
import pdb
import pylab
from matplotlib import pyplot as plt

class CTEnv(Env):
    def __init__(self):
        # Image / sinogram dimensions
        self.numpix = 512     #image size
        self.numtheta = 90    #90 views for sparse-view simulation
	self.numbin = 729     #number of projection bins
        self.dx = 0.0568      #assumed pixel size

        #fan beam parameters
	self.geom = 'fanflat'		#could be parallel as well
	self.dso, self.dod  = 100, 100 #source-object and object-detector distances
	self.fan_angle = 35		#fanbeam angle
	self.P, self.D, self.M = projNorms()   #projection operation and SART matrices
	
	#sinogram and true image (used to compute reward)
	self.sino = np.fromfile('sinogram_name.flt',dtype='f')
	self.sino = self.sino.reshape(self.numtheta,self.numbin)
	self.x_true = np.fromfile('true_image_name.flt',dtype='f')
	self.x_true = self.x_true.reshape(self.numpix,self.numpix)
        self.state = np.zeros((numpix,numpix))   #initial image (zeroes)

        # Actions: SART (0) or Superiorization (1)
        self.action_space = Discrete(2)

        # Image
        self.observation_space = Box(low=np.array([0]), high=np.inf, shape=(numpix,numpix))

        # Algorithm parameters
	self.ns = 1
        self.num_its = 0
	self.max_iters = 100
        self.alpha = 1 # makes size of pertubation go down
	self.gamma = 0.995
        self.beta = 1
	self.epsilon_target = 0;

    def step(self, action):
        f = self.state 
        # Calculate reward
        if (action == 0): # SARTi
	    self.num_its = self.num_its+1
            for j in range(self.ns): # copied from SART.py
                ind1 = range(j,self.numtheta,self.ns);
                p = self.P[j]
                fp_id, fp = astra.create_sino(f,p)      #forward projection step
                diffs = (self.sino[ind1,:] - fp*self.dx)/self.Minv[j]                  #should do elementwise division
                bp_id, bp = astra.create_backprojection(diffs,p)
                ind2 = np.abs(bp) > 1e3
                bp[ind2] = 0             #get rid of spurious large values
                f = f + self.beta * bp/self.Dinv[j];                   #update f
                astra.data2d.delete(fp_id)
                astra.data2d.delete(bp_id)
        elif (action == 1): # Superiorization
            #pdb.set_trace()
            g,dg = grad_TV(f,self.numpix)
            g_old = g;
            dg = -dg / (np.linalg.norm(dg,'fro') + eps)
            # for j in range(N):
            while True:
                f_tmp = f + self.alpha * dg
                g_new = grad_TV(f_tmp,self.numpix)[0]
                self.alpha = self.alpha*self.gamma
                if g_new <= g_old:
                    f = f_tmp
                    break

        # Reward
        # - need a way to load in x_true (true image) to compute the reward
        #   - load in binary files (.flt) to compare / computing the reward
        reward = np.linalg.norm(f, 'fro') / np.linalg.norm((f - self.x_true), 'fro')

        # Check if code is done
        fp = np.zeros((self.numtheta,self.numbin)) # computes (||Ax-b||)
        for j in range(self.ns):
            ind = range(j,numtheta,self.ns)
            p = self.P[j]
            fp_tempid, fp_temp = astra.create_sino(f,p)
            fp[ind,:] = fp_temp*self.dx
            astra.data2d.delete(fp_tempid)

        res = np.linalg.norm(fp-self.sino,'fro')

        if ((self.num_its >= self.max_iters) or (res < self.epsilon_target)):
            done = True
        
        # Return step information
        self.state = f
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.state = np.zeros((numpix,numpix))
        self.num_its = 0
        self.alpha = 1

    ##         ## 
    ## HELPERS ##
    ##         ## 

    def create_projector():
        if self.geom == 'parallel':
            proj_geom = astra.create_proj_geom(self.geom, 1.0, self.numbin, self.numtheta)
        elif self.geom == 'fanflat':
            self.dso *=10; self.dod *=10;                         #convert to mm for astra
            ft = np.tan( np.deg2rad(self.fan_angle / 2) )    #compute tan of 1/2 the fan angle
            det_width = 2 * (self.dso + self.dod) * ft / self.numbin  #width of one detector pixel, calculated based on fan angle
	    vol_geom = astra.create_vol_geom(self.numpix,self.numpix)
            proj_geom = astra.create_proj_geom(self.geom, det_width, self.numbin, self.numtheta, self.dso, self.dod)

        p = astra.create_projector('cuda',proj_geom,vol_geom);
        return p

    def grad_TV(img,numpix):
        #pdb.set_trace()
        epsilon = 1e-6
        ind_m1 = np.arange(numpix)
        ind_m2 = [(i + 1) % numpix for i in ind_m1]
        ind_m0 = [(i - 1) % numpix for i in ind_m1]

        m2m1 = np.ix_(ind_m2,ind_m1)
        m1m2 = np.ix_(ind_m1,ind_m2)
        m0m1 = np.ix_(ind_m0,ind_m1)
        m1m0 = np.ix_(ind_m1,ind_m0)
        
        diff1 = ( img[m2m1] - img) ** 2
        diff2 = ( img[m1m2] - img) ** 2
        diffttl = np.sqrt( diff1 + diff2 + epsilon**2)
        TV = np.sum(diffttl)

        dTV = -1/diffttl * (img[m2m1]-2*img + img[m1m2]) + \
            1/diffttl[m0m1] * (img-img[m0m1]) + \
            1/diffttl[m1m0] * (img-img[m1m0])
        return TV, dTV

    #create projectors and normalization terms (corresponding to diagonal matrices M and D) for each subset of projection data
    def projNorms():
        P, Dinv, D_id, Minv, M_id = [None]*ns,[None]*ns,[None]*ns,[None]*ns,[None]*ns
        for j in range(ns):
            ind1 = range(j,numtheta,ns);
            p = create_projector(geom,numbin,angles[ind1],dso,dod,fan_angle)
            
            D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta//ns,numbin)),p)
            M_id[j], Minv[j] = astra.create_sino(np.ones((numpix,numpix)),p)
            #avoid division by zero, also scale M to pixel size
            Dinv[j] = np.maximum(Dinv[j],eps)
            Minv[j] = np.maximum(Minv[j]*dx,eps)
            P[j] = p

	return P, D, M
