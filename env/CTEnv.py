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
        # Variables
        self.numpix = 512
        self.ns = 1
        self.numtheta = 900
        #self.sino
        self.dx = 1
        self.Minv
        self.Dinv
        self.beta = 1
        # self.eps
        # self.gamma

        # Actions: SART (0) or Superiorization (1)
        self.action_space = Discrete(2)

        # Image
        self.observation_space = Box(low=np.array([0]), high=np.inf, shape=(numpix,numpix))

        # Set start
        self.state = np.zeros((numpix,numpix))
        num_its = 0
        alpha = 1 # makes size of putubation go down
        
    def step(self, action):
        f = self.state 

        # Calculate reward
        if (action == 0): # SART
            for j in range(ns): # copied from SART.py
                ind1 = range(j,numtheta,ns);
                p = P[j]
                fp_id, fp = astra.create_sino(f,p)      #forward projection step
                diffs = (sino[ind1,:] - fp*dx)/Minv[j]                  #should do elementwise division
                bp_id, bp = astra.create_backprojection(diffs,p)
                ind2 = np.abs(bp) > 1e3
                bp[ind2] = 0             #get rid of spurious large values
                f = f + beta * bp/Dinv[j];                   #update f
                astra.data2d.delete(fp_id)
                astra.data2d.delete(bp_id)
        elif (action == 1): # Superiorization
            #pdb.set_trace()
            g,dg = grad_TV(f,numpix)
            g_old = g;
            dg = -dg / (np.linalg.norm(dg,'fro') + eps)
            # for j in range(N):
            while True:
                f_tmp = f + alpha * dg
                g_new = grad_TV(f_tmp,numpix)[0]
                alpha = alpha*gamma
                if g_new <= g_old:
                    f = f_tmp
                    break

        # Reward
        # - need a way to load in x_true (true image) to compute the reward
        #   - load in binary files (.flt) to compare / computing the reward
        reward = np.linalg.norm(f, 'fro') / np.linalg.norm((f - x_true), 'fro')

        # Check if code is done
        fp = np.zeros((numtheta,numbin)) # computes (||Ax-b||)
        for j in range(ns):
            ind = range(j,numtheta,ns)
            p = P[j]
            fp_tempid, fp_temp = astra.create_sino(f,p)
            fp[ind,:] = fp_temp*dx
            astra.data2d.delete(fp_tempid)

        res = np.linalg.norm(fp-sino,'fro') / np.linalg.norm(sino,'fro')

        if ((k >= max_iters) or (res < epsilon)):
            done = True
        
        # Return step information
        self.state = f
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        self.state = np.zeros((numpix,numpix))
        num_its = 0
        alpha = 1

    ##         ## 
    ## HELPERS ##
    ##         ## 

    def create_projector(geom, numbin, angles, dso, dod, fan_angle):
        if geom == 'parallel':
            proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
        elif geom == 'fanflat':
            dso *=10; dod *=10;                         #convert to mm for astra
            ft = np.tan( np.deg2rad(fan_angle / 2) )    #compute tan of 1/2 the fan angle
            det_width = 2 * (dso + dod) * ft / numbin  #width of one detector pixel, calculated based on fan angle

            proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)

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