#!/usr/bin/env python
#
# sudo apt install libopencv-dev python-opencv cmake libboost-all-dev
# sudo pip install facemorpher docopt future dlib
#
# TODO:
# Try the dlib facial feature detector:
#   http://dlib.net/face_landmark_detection.py.html
#   http://www.codesofinterest.com/2016/10/getting-dlib-face-landmark-detection.html
#

import facemorpher as fm
from glob import glob
import os
import numpy as np

imdir = '/home/bobd/git/mindstrong/face_morpher/nimFaces/'
outdir = '/home/bobd/git/mindstrong/neuropsych/experiments/face_morph/images_raw'
if not os.path.exists(outdir):
    os.makedirs(outdir)

w = 500
h = 600
nframes = 52
emotions = ['ha','sa','an']#,'fe','di']
mouth = 'c'
col_nums = [5,7,1,4,3]
val = np.genfromtxt(os.path.join(imdir,'validity.csv'), delimiter=',',skip_header=1, missing_values=' ')
rel = np.genfromtxt(os.path.join(imdir,'reliability.csv'), delimiter=',',skip_header=1, missing_values=' ')
sc = np.hstack((val[:,0][:,None],val[:,1:] * rel[:,1:]))[:,[0]+col_nums]
# Remove actors who don't have all our emotions:
sc = sc[np.all(np.isnan(sc)==False,axis=1),:]
sc_sum = sc[:,1:].mean(axis=1)
models = sc[:,0].astype(np.int)

for m in models:
    for em in emotions:
        s = 'f' if m<20 else 'm'
        mstr = '%02d%s' % (m,s)
        outvid = None
        outfr = os.path.join(outdir, mstr+'_'+em+'_'+mouth)
        imgpaths = [os.path.join(imdir, 'stim', mstr+'_ne_'+mouth+'.png'),
                    os.path.join(imdir, 'stim', mstr+'_'+em+'_'+mouth+'.png')]
        if os.path.exists(imgpaths[0]) and os.path.exists(imgpaths[1]):
            fm.morpher(imgpaths, alpha=False, width=w, height=h, num_frames=nframes, fps=10, out_frames=outfr, out_video=outvid)
        else:
            print('missing %s/%s' % (imgpaths[0],imgpaths[1]))



# Creating an average face
for em in emotions:
    imgpaths = glob(os.path.join(imdir, 'stim', '*f_'+em+'_c.png'))
    fm.averager(imgpaths, alpha=True, blur_edges=True, out_filename='female_'+em+'_c.png')


import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import scipy
impath = imgpaths[0]
img = scipy.ndimage.imread(impath)[..., :3]
points = fm.locator.face_points(impath)
plt.imshow(img)
plt.plot(points[:,0], points[:,1], '.')


