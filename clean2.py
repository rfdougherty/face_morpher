#!/usr/bin/env python

# TODO:
# * load all images and compute a common bounding box
# * load them all again, clip to the common bb, resize to ~ 300px height, flood-fill the black
#   boarder voxels with white (or alpha=1), save as jpeg with ~0.75 compression.

import numpy as np
import os
# Need the scikit-image module
from skimage import io, transform
from glob import glob
from scipy import ndimage
from skimage.util.montage import montage2d
from subprocess import call
from tempfile import mkdtemp
from shutil import rmtree

def get_mask(img, pad=0):
    '''Compute a mask to within pad voxels of non-zero data.'''
    im = img[...,:3].mean(axis=2)
    xlim = np.argwhere(im.sum(axis=1))[[0,-1]].squeeze()
    xlim = np.hstack((np.max((0,xlim[0]-pad)), np.min((im.shape[0],xlim[1]+pad))))
    ylim = np.argwhere(im.sum(axis=0))[[0,-1]].squeeze()
    ylim = np.hstack((np.max((0,ylim[0]-pad)),np.min((im.shape[1],ylim[1]+pad))))
    mask = np.zeros(im.shape, dtype=np.uint16)
    mask[xlim[0]:xlim[1], ylim[0]:ylim[1]] = 1
    return mask

def crop(img, mask, pad=0):
    '''Crop a numpy array to within pad voxels of non-zero mask.'''
    xlim = np.argwhere(mask.sum(axis=1))[[0,-1]].squeeze()
    xlim = np.hstack((np.max((0,xlim[0]-pad)), np.min((mask.shape[0],xlim[1]+pad))))
    ylim = np.argwhere(mask.sum(axis=0))[[0,-1]].squeeze()
    ylim = np.hstack((np.max((0,ylim[0]-pad)),np.min((mask.shape[1],ylim[1]+pad))))
    img =  img[xlim[0]:xlim[1], ylim[0]:ylim[1], :]
    return img


imdir = '/Users/bobd/git/face_morpher/p1vital/out/'
dirs = glob(os.path.join(imdir, '*'))

outbase = '/Users/bobd/git/face_morpher/p1vital/cleaned/'

if not os.path.exists(outbase):
    out_sz = (300,240)
    all_mask = 0
    print('Generating mask...')
    for d in dirs:
        print('  %s...' % os.path.basename(d))
        frames = glob(os.path.join(d,'*.png'))
        for f in frames:
            img =  io.imread(f)
            all_mask += get_mask(img)

    print('Cropping images...')
    for d in dirs:
        dirname = os.path.basename(d)
        print('  %s...' % dirname)
        frames = glob(os.path.join(d,'*.png'))
        outdir = os.path.join(outbase, os.path.basename(d))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for f in frames:
            img = crop(io.imread(f), all_mask, pad=-10)
            img = transform.resize(img, out_sz, order=5)
            bn = os.path.basename(f)[:-3]
            io.imsave(os.path.join(outdir, bn+'png'), img)
            if bn=='frame001.':
                io.imsave(os.path.join(outdir, 'neutral.jpg'), img)
            elif bn=='frame050.':
                io.imsave(os.path.join(outdir, dirname+'.jpg'), img)

dirs=glob(os.path.join(outbase,'*'))
for d in dirs:
    dirname = os.path.basename(d)
    print('ffmpeg  %s...' % dirname)
    call(['ffmpeg','-framerate','5','-i',os.path.join(d,'frame%03d.png'),'-c:v','libx264','-profile:v','baseline','-preset','veryslow','-crf','24','-vf','fps=5','-pix_fmt', 'yuv420p','-movflags','+faststart',os.path.join(d,dirname+'.mp4')])


#dirs=glob('*')
#for d in dirs: os.rename(d+'/neutral.jpg',d[:4]+'ne'+d[6:]+'.jpg')

