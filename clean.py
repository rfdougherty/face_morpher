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
#from skimage.filter import sobel
#from skimage.morphology import watershed
from scipy import ndimage
#from skimage.util.montage import montage2d
from PIL import Image

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


imdir = '/home/bobd/git/mindstrong/neuropsych/experiments/face_morph/images_raw/'
dirs = glob(os.path.join(imdir, '*_c'))

outbase = '/home/bobd/git/mindstrong/neuropsych/experiments/face_morph/cleaned_webp/'

out_sz = (250,200)
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
    print('  %s...' % os.path.basename(d))
    frames = glob(os.path.join(d,'*.png'))
    outdir = os.path.join(outbase, os.path.basename(d))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for f in frames:
        img = crop(io.imread(f), all_mask, pad=-10)
        img = transform.resize(img, out_sz, order=5)
        #io.imsave(os.path.join(outdir, os.path.basename(f)[:-3]+'jpg'), img)
        im = Image.fromarray((img*255).round().astype(np.uint8), 'RGBA')
        im.save(os.path.join(outdir, os.path.basename(f)[:-3]+'webp'), "WEBP", quality=75)

