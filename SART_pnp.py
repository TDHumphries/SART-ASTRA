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
#import pylab
from PIL import Image
import tensorflow as tf
from model import denoiser
from utils import *

def create_projector(geom, numbin, angles, dso, dod, fan_angle):
    if geom == 'parallel':
        proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
    elif geom == 'fanflat':
        dso *=10; dod *=10;                         #convert to mm for astra
        ft = np.tan( np.deg2rad(fan_angle / 2) )    #compute tan of 1/2 the fan angle
        det_width = 2 * (dso + dod) * ft / numbin  #width of one detector pixel, calculated based on fan angle

        proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)

    p = astra.create_projector('strip_fanflat',proj_geom,vol_geom);
    return p


parser = argparse.ArgumentParser(description='')
parser.add_argument('--sino', dest='infile', default='.', help='input sinogram -- directory or single file')
parser.add_argument('--out', dest='outfile', default='.', help='output directory')
parser.add_argument('--numpix',dest='numpix',type=int,default=512,help='size of volume (n x n )')
parser.add_argument('--psize',dest='psize',default='',help='pixel size (float) OR file containing pixel sizes (string)');
parser.add_argument('--numbin',dest='numbin',type=int,default=729,help='number of detector pixels')
parser.add_argument('--ntheta',dest='numtheta',type=int,default=900,help='number of angles')
parser.add_argument('--nsubs',dest='ns',type=int,default=1,help='number of subsets. must divide evenly into number of angles')
parser.add_argument('--range', dest='theta_range',type=float,nargs=2,default=[0, 360],help='starting and ending angles (deg)')
parser.add_argument('--geom', dest='geom',default='fanflat',help='geometry (parallel or fanflat)')
parser.add_argument('--dso',dest='dso',type=float,default=100,help='source-object distance (cm) (fanbeam only)')
parser.add_argument('--dod',dest='dod',type=float,default=100,help='detector-object distance (cm) (fanbeam only)')
parser.add_argument('--fan_angle',dest='fan_angle',default=35,type=float,help='fan angle (deg) (fanbeam only)')
parser.add_argument('--numits',dest='num_its',default=32,type=int,help='maximum number of iterations')
parser.add_argument('--beta',dest='beta',default=1.,type=float,help='relaxation parameter beta')
parser.add_argument('--x0',dest='x0_file',default='',help='initial image (default: zeros)')
parser.add_argument('--xtrue',dest='xtrue_file',default='',help='true image (if available)')
parser.add_argument('--sup_params',dest='sup_params', type=float,nargs=3,help='superiorization parameters gamma, N, alpha_init')
parser.add_argument('--epsilon_target',dest='epsilon_target',default=0.,help='target residual value to stop (float or file with residual values)')
parser.add_argument('--ckpt_dir',dest='ckpt_dir',default='./checkpoint',help='directory containing checkpoint for DnCnn')

#get arguments from command line
args = parser.parse_args()
infile, outfile, x0file, xtruefile, psize = args.infile, args.outfile, args.x0_file, args.xtrue_file, args.psize
numpix, numbin, numtheta, ns, numits, beta, epsilon_target = args.numpix, args.numbin, args.numtheta, args.ns, args.num_its, args.beta, args.epsilon_target
theta_range, geom, dso, dod, fan_angle = args.theta_range, args.geom, args.dso, args.dod, args.fan_angle

if args.sup_params is None:
    use_sup = False
else:
    use_sup = True
          
eps = np.finfo(float).eps

if os.path.isdir(infile):		#generate list of filenames from directory
    fnames = sorted(glob(infile + '/*.flt'))
else:							#single filename
    fnames = []
    fnames.append(infile)
#fnames = fnames[204:]

try:
        psizes = float(psize)      #if pixel size is a floating point value
except ValueError:
        psizes = np.loadtxt(psize,dtype='f')            #if pixel size is a file name

try:
    epsilon_target = float(epsilon_target)
except ValueError:
    epsilon_target = np.loadtxt(epsilon_target,dtype='f')
    res_counter = 0

# create projection geometry
vol_geom = astra.create_vol_geom(numpix, numpix)

#generate array of angular positions
theta_range = np.deg2rad(theta_range) #convert to radians
angles = theta_range[0] + np.linspace(0,numtheta-1,numtheta,False)*(theta_range[1]-theta_range[0])/numtheta #

#if x0file == '':        
#    f = np.zeros((numpix,numpix))
#else:
#    f = np.fromfile(x0file,dtype='f')
#    f = f.reshape(numpix,numpix)

#if xtruefile == '':
#    calc_error = False
#    xtrue = np.zeros((numpix,numpix))
#else:
#    xtrue = np.fromfile(xtruefile,dtype='f')
#    xtrue = xtrue.reshape(numpix,numpix)
#    calc_error = True

calc_error = False
    
#create projectors and normalization terms (corresponding to diagonal matrices M and D) for each subset of projection data
P, Dinv, D_id, Minv, M_id = [None]*ns,[None]*ns,[None]*ns,[None]*ns,[None]*ns
for j in range(ns):
    ind1 = range(j,numtheta,ns);
    p = create_projector(geom,numbin,angles[ind1],dso,dod,fan_angle)
    
    D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta//ns,numbin)),p)
    M_id[j], Minv[j] = astra.create_sino(np.ones((numpix,numpix)),p)
    #avoid division by zero, also scale M to pixel size
    Dinv[j] = np.maximum(Dinv[j],eps)
    Minv[j] = np.maximum(Minv[j],eps)
    P[j] = p
res_file = open(outfile+"/residuals.txt","w+") #file to store residuals

if use_sup:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)    
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        model = denoiser(sess)
        tf.initialize_all_variables().run(session=model.sess)
        load_model_status, global_step = model.load(args.ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

for name in fnames:
#    pdb.set_trace()
    #read in sinogram
    sino = np.fromfile(name,dtype='f')
    sino = sino.reshape(numtheta,numbin)

    head, tail = os.path.split(name)      #get name of file for output
    head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_sino.flt
    f = np.zeros((numpix,numpix))
    try:
        dx = psizes[int(head)] #if psize being read in from file
    except:
        dx = psizes 	#if psize is a float

    try:
        etarget = epsilon_target[res_counter]
        res_counter+=1
    except:
        etarget = epsilon_target

    if use_sup:                 #reinitialize superiorization parameters for every new image
        gamma = args.sup_params[0]
        N = np.uint8(args.sup_params[1])
        alpha = args.sup_params[2]
    
    for k in range(numits):
        #Superiorization loop
        if use_sup:
            f_out = model.run(f)
            #pdb.set_trace()
            p = f_out - f
            pnorm = np.linalg.norm(p,'fro')+eps
            if pnorm > alpha:
                p = alpha*p/ (np.linalg.norm(p,'fro')+eps)
                f = f + p
            else:
                f = f_out
            alpha = alpha*gamma
            #img = (f.T/np.amax(f)) * 255
            #img = np.round(img)
            #im = Image.fromarray(img.astype('uint8')).convert('L')
            #im.save(outname+'_'+str(k)+'sup.png','png')
 

        #SART loop
        for j in range(ns):
            ind1 = range(j,numtheta,ns);
            p = P[j]
            fp_id, fp = astra.create_sino(f,p)      #forward projection step
            diffs = (sino[ind1,:] - fp*dx)/Minv[j]/dx                  #should do elementwise division
            bp_id, bp = astra.create_backprojection(diffs,p)
            ind2 = np.abs(bp) > 1e3
            bp[ind2] = 0             #get rid of spurious large values
            f = f + beta * bp/Dinv[j];                   #update f
    #        pdb.set_trace()
            astra.data2d.delete(fp_id)
            astra.data2d.delete(bp_id)
            
        f = np.maximum(f,eps);                  #set any negative values to small positive value
        #img = (f.T/np.amax(f)) * 255
        #img = np.round(img)
        #im = Image.fromarray(img.astype('uint8')).convert('L')
        #im.save(outname+'_'+str(k)+'SART.png','png')
        
        #compute residual and error (if xtrue is provided)
        fp = np.zeros((numtheta,numbin))
        for j in range(ns):
            ind = range(j,numtheta,ns)
            p = P[j]
            fp_tempid, fp_temp = astra.create_sino(f,p)
            fp[ind,:] = fp_temp*dx
            astra.data2d.delete(fp_tempid)

        res = np.linalg.norm(fp-sino,'fro')
        if (res < etarget):
            break
        #pdb.set_trace()
        if calc_error:
             err = np.linalg.norm(f-xtrue,'fro')/np.linalg.norm(xtrue,'fro')
             print('Iteration #{0:d}: Residual = {1:1.4f}\tError = {2:1.4f}\n'.format(k+1,res,err))
        else:
             print('Iteration #{0:d}: Residual = {1:1.4f}\n'.format(k+1,res))
#        if (k == 0 or k == 2 or k == 5):
 #           ft = np.float32(f)
 #           ft.tofile(outfile + '/'+head+'_recon_'+str(k+1)+'.flt') 
    #save image
    f = np.float32(f)
    outname = outfile + "/" + head + "_recon_"+str(k)+".flt"
    f.tofile(outname)

    #save residual
    res_file.write("%f\n" % res)

    #**********save image as png**********
    max_pixel = np.amax(f)
    img = (f.T/max_pixel) * 255
    img = np.round(img)
    im = Image.fromarray(img.astype('uint8')).convert('L')
    im.save(outname+'.png','png')
    # plt.figure(1)
    # plt.style.use('grayscale')
    # plt.imshow(img.T) #transpose image
    # plt.axis('off')
    # png_outname = (outname + '.png')
    # plt.savefig(png_outname)
    # plt.close()
    #**************************************

res_file.close()
for j in range(ns):
    astra.data2d.delete(D_id[j])
    astra.data2d.delete(M_id[j])
    astra.projector.delete(P[j])
