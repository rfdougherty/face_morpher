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

imdir = '/home/bobd/git/mindstrong/nimFaces/stim/'

w = 500
h = 600
imgpaths = [imdir+'17f_ne_c.png',imdir+'17f_ha_c.png']
nframes = 50
emotions = ['ha','sa','an','fe','di']
mouth = 'c'
col_nums = [5,7,1,4,3]
val = np.genfromtxt('../nimFaces/validity.csv', delimiter=',',skip_header=1, missing_values=' ')
rel = np.genfromtxt('../nimFaces/reliability.csv', delimiter=',',skip_header=1, missing_values=' ')
sc = np.hstack((val[:,0][:,None],val[:,1:] * rel[:,1:]))[:,[0]+col_nums]
# Remove actors who don't have all our emotions:
sc = sc[np.all(np.isnan(sc)==False,axis=1),:]
sc_sum = sc[:,1:].mean(axis=1)
models = sc[:,0].astype(np.int)

for m in models:
    for em in emotions:
        mstr = '%02d' % m
        outvid = None
        outfr = mstr+'_'+em+'_'+mouth
        s = 'f' if m<20 else 'm'
        imgpaths = [os.path.join(imdir, mstr+s+'_ne_'+mouth+'.png'),
                    os.path.join(imdir, mstr+s+'_'+em+'_'+mouth+'.png')]
        fm.morpher(imgpaths, alpha=False, width=w, height=h, num_frames=nframes, fps=10, out_frames=outfr, out_video=outvid)


# Get a list of image paths in a folder
#imgpaths = facemorpher.list_imgpaths('stim')
imgpaths = glob(imdir+'*f_ne_o.png')
fm.averager(imgpaths, alpha=True, blur_edges=True, out_filename='female_ha_o.png', plot=True)


import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import scipy
impath = imgpaths[0]
img = scipy.ndimage.imread(impath)[..., :3]
points = fm.locator.face_points(impath)
plt.imshow(img)
plt.plot(points[:,0], points[:,1], '.')


# Create a numpy array of floats to store the average (assume RGB images)
arr = np.zeros((h,w,3),np.float)

# Build up average pixel intensities, casting each image as an array of floats
for im in imgpaths:
    imarr = np.array(Image.open(im),dtype=np.float)
    arr = arr+imarr/N

# Round values in array and cast as 8-bit integer
arr = np.array(np.round(arr),dtype=np.uint8)

# Generate, save and preview final image
out = Image.fromarray(arr,mode="RGB")
#out.save("Average.png")
out.show()


out_video = '17_ha.avi'
fps=10
num_frames = nframes
out_frames = None
plot = True
width = 500
height = 600

video = facemorpher.videoer.Video(out_video, fps, width, height)
images_points_gen = facemorpher.load_valid_image_points(imgpaths, (height, width))
src_img, src_points = next(images_points_gen)
for dest_img, dest_points in images_points_gen:
    facemorpher.morph(src_img, src_points, dest_img, dest_points, video,
          width, height, num_frames, fps, out_frames, out_video, alpha, plot)
    src_img, src_points = dest_img, dest_points
video.end()

