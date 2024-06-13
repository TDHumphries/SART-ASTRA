# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
import argparse
from glob import glob
import astra
import numpy as np
import os
import pdb
import pylab
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt # for png conversion

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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in', dest='infile', default='.', help='input image -- directory or single file')
parser.add_argument('--out', dest='outfile', default='.', help='output directory')
parser.add_argument('--numpix',dest='numpix',type=int,default=512,help='size of volume (n x n )')
parser.add_argument('--psize',dest='psize',default='',help='pixel size (float) OR file containing pixel sizes (string)');
parser.add_argument('--numbin',dest='numbin',type=int,default=729,help='number of detector pixels')
parser.add_argument('--ntheta',dest='numtheta',type=int,default=900,help='number of angles')
parser.add_argument('--range', dest='theta_range',type=float,nargs=2,default=[0, 360],help='starting and ending angles (deg)')
parser.add_argument('--geom', dest='geom',default='fanflat',help='geometry (parallel or fanflat)')
parser.add_argument('--dso',dest='dso',type=float,default=100,help='source-object distance (cm) (fanbeam only)')
parser.add_argument('--dod',dest='dod',type=float,default=100,help='detector-object distance (cm) (fanbeam only)')
parser.add_argument('--fan_angle',dest='fan_angle',default=35,type=float,help='fan angle (deg) (fanbeam only)')

#get arguments from command line
args = parser.parse_args()
infile, outfile, psize = args.infile, args.outfile, args.psize
numpix, numbin, numtheta = args.numpix, args.numbin, args.numtheta
theta_range, geom, dso, dod, fan_angle = args.theta_range, args.geom, args.dso, args.dod, args.fan_angle
          
eps = np.finfo(float).eps
counts = 1e6

if os.path.isdir(infile):		#generate list of filenames from directory
    fnames = sorted(glob(infile + '/*.flt'))
else:							#single filename
    fnames = []
    fnames.append(infile)

try:
        psizes = float(psize)      #if pixel size is a floating point value
except ValueError:
        psizes = np.loadtxt(psize,dtype='f')            #if pixel size is a file name

# create projection geometry
vol_geom = astra.create_vol_geom(numpix, numpix)

#generate array of angular positions
theta_range = np.deg2rad(theta_range) #convert to radians
angles = theta_range[0] + np.linspace(0,numtheta-1,numtheta,False)*(theta_range[1]-theta_range[0])/numtheta #

#create projectors and normalization terms (corresponding to diagonal matrices M and D) for each subset of projection data
p = create_projector(geom,numbin,angles,dso,dod,fan_angle)
pdb.set_trace()
for name in fnames:
    #read in image
    img = np.fromfile(name,dtype='f')
    img = img.reshape(numpix,numpix)

    head, tail = os.path.split(name)      #get name of file for output
    head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_sino.flt
    outname = outfile + "/" + head + "_recon_FBP.flt"
    try:
        dx = psizes[int(head)] #if psize being read in from file
    except:
        dx = psizes 	#if psize is a float
        
    [sino_id,sino] = astra.create_sino(img,p)
    sino = sino * dx                #normalize to pixel size
    sino = counts * np.exp(-sino)   #exponentiate
    sino = np.random.poisson(sino)  #add noise
    sino = -np.log(sino/counts)   #return to log domain
    recon_id = astra.data2d.create('-vol',vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sino_id
    cfg['FilterType'] = 'Ram-Lak'

    fbp_id = astra.algorithm.create(cfg)
    astra.algorithm.run(fbp_id)
    f = astra.data2d.get(recon_id)/dx   
    #save image
    f = np.float32(f)
    f.tofile(outname)

    #save residual

    #**********save image as png**********
    max_pixel = np.amax(f)
    img = (f/max_pixel) * 255
    img = np.round(img)

    plt.figure(1)
    plt.style.use('grayscale')
    plt.imshow(img.T) #transpose image
    plt.axis('off')
    png_outname = (outname + '.png')
    plt.savefig(png_outname)
    plt.close()
    #**************************************
    astra.algorithm.delete(fbp_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sino_id)
    
astra.projector.delete(p)
