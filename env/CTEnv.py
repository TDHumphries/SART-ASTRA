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

# ##         ## 
# ## HELPERS ##
# ##         ## 

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
def projNorms(ns, numtheta, geom, numbin, angles, dso, dod, fan_angle, numpix, dx):
	P, Dinv, D_id, Minv, M_id = [None]*ns,[None]*ns,[None]*ns,[None]*ns,[None]*ns
	eps = np.finfo(float).eps

	for j in range(ns):
		ind1 = range(j,numtheta, ns);
		p = create_projector(geom,numbin,angles[ind1],dso,dod,fan_angle,numpix)
		
		D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta//ns,numbin)),p)
		M_id[j], Minv[j] = astra.create_sino(np.ones((numpix,numpix)),p)
		#avoid division by zero, also scale M to pixel size
		Dinv[j] = np.maximum(Dinv[j],eps)
		Minv[j] = np.maximum(Minv[j]*dx,eps)
		P[j] = p

	return P, Dinv, Minv

def create_projector(geom, numbin, angles, dso, dod, fan_angle, numpix):
	if geom == 'parallel':
		proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
	elif geom == 'fanflat':
		dso *=10; dod *=10;                    #convert to mm for astra
		ft = np.tan( np.deg2rad(fan_angle / 2) )    #compute tan of 1/2 the fan angle
		det_width = 2 * (dso + dod) * ft / numbin  #width of one detector pixel, calculated based on fan angle
		vol_geom = astra.create_vol_geom(numpix,numpix)
		proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)

	p = astra.create_projector('cuda',proj_geom,vol_geom);
	return p

class CTEnv(Env):
	def __init__(self):
		super(CTEnv, self).__init__()
		# Image / sinogram dimensions
		self.numpix = 512     #image size
		self.numtheta = 90    #900 views for sparse-view simulation
		self.numbin = 729     #number of projection bins
		self.dx = 0.0568      #assumed pixel size
		self.angles = np.linspace(0,self.numtheta-1,self.numtheta)*360/self.numtheta
		self.angles = np.deg2rad(self.angles)

		#fan beam parameters
		self.geom = 'fanflat'		#could be parallel as well
		self.dso, self.dod  = 100, 100 #source-object and object-detector distances
		self.fan_angle = 35		#fanbeam angle

		self.sino_file = open("sinos/00000001_sino.flt")
		self.x_true_file = open("imgs/00000001_img.flt")
	
		#sinogram and true image (used to compute reward)
		self.sino = np.fromfile(self.sino_file,dtype='f') # in sinos folder
		self.sino = self.sino.reshape(self.numtheta,self.numbin)
		self.x_true = np.fromfile(self.x_true_file,dtype='f') # file imgs folder
		self.x_true = self.x_true.reshape(self.numpix,self.numpix)
		self.state = np.zeros((self.numpix,self.numpix))   #initial image (zeroes)

		# Actions: SART (0) or Superiorization (1)
		self.action_space = Discrete(2)

		# Image
		# self.observation_space = Box(low=np.array([0]), high=np.array([self.numpix]), shape=(1,15))
		self.observation_space = Box(low=0, high=np.inf, shape=(self.numpix,self.numpix),dtype=np.float32)

		# Algorithm parameters
		self.ns = 10
		self.num_its = 0
		self.max_iters = 40
		self.alpha = 1 # makes size of pertubation go down
		self.gamma = 0.995
		self.beta = 1
		self.epsilon_target = 0
		self.episode = 0

		#projection operation and SART matrices
		self.P, self.Dinv, self.Minv = projNorms(self.ns, self.numtheta, self.geom, self.numbin, self.angles, self.dso, self.dod, self.fan_angle, self.numpix, self.dx)   

	def step(self, action):
		f = self.state
		#pdb.set_trace()
		# Calculate reward
		if (action == 0): # SART
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
			eps = np.finfo(float).eps # prevent division by zero
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
		f = np.maximum(f,np.finfo(float).eps)
		reward = np.linalg.norm(f, 'fro') / np.linalg.norm((f - self.x_true), 'fro')

		# Check if code is done
		fp = np.zeros((self.numtheta,self.numbin)) # computes (||Ax-b||)
		for j in range(self.ns):
			ind = range(j,self.numtheta,self.ns)
			p = self.P[j]
			fp_tempid, fp_temp = astra.create_sino(f,p)
			fp[ind,:] = fp_temp*self.dx
			astra.data2d.delete(fp_tempid)

		res = np.linalg.norm(fp-self.sino,'fro')

		if ((self.num_its >= self.max_iters) or (res < self.epsilon_target)):
			done = True
			self.render()
			self.episode = self.episode+1
		else:
			done = False
		
		# Return step information
		self.state = f
		
		return self.state, reward, done, {}

	def render(self):
		#pdb.set_trace()
		plt.figure(num=1)
		plt.style.use('grayscale')
		plt.imshow(self.state,vmin = 0, vmax = 0.4)
		plt.colorbar()
		plt.savefig('training_imgs/tmp{0:d}.png'.format(self.episode))
		plt.close()               
			
	def reset(self):
		self.state = np.zeros((self.numpix,self.numpix))
		self.num_its = 0
		self.alpha = 1
		#self.observation_space = Box(low=0, high=np.inf, shape=(self.numpix,self.numpix))

		#return self.observation_space
		return self.state
