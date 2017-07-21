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

from glob import glob
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np

basedir = '/Users/bobd/git/face_morpher'
os.chdir(os.path.join(basedir,'facemorpher'))
import facemorpher as fm
os.chdir(basedir)

imdir = os.path.join(basedir, 'KDEF', 'raw')
outdir = os.path.join(basedir, 'KDEF', 'out')
if not os.path.exists(outdir):
    os.makedirs(outdir)

w = 500
h = 600
nframes = 52
versions = [('A','A'),('A','B'),('B','A'),('B','B')]
ne = 'NE'
suffix = 'S.JPG'
emos = ['HA','SA','AN','AF','DI','SU']
models = [s+'%02d'%m for m in range(1, 36) for s in ['M','F']]

for v in versions:
    for m in models:
        for em in emos:
            outfr = os.path.join(outdir, ''.join(v)+m+'_'+em)
            if os.path.exists(outfr) and len(glob(os.path.join(outfr,'*')))>=50:
                print('%s is already done...' % outfr)
            else:
                imgpaths = [os.path.join(imdir, v[0]+m, v[0]+m+ne+suffix),
                            os.path.join(imdir, v[1]+m, v[1]+m+em+suffix)]
                if os.path.exists(imgpaths[0]) and os.path.exists(imgpaths[1]):
                    fm.morpher(imgpaths, alpha=True, width=w, height=h, num_frames=nframes, fps=10, out_frames=outfr, out_video=None)
                else:
                    print('missing %s/%s' % (imgpaths[0],imgpaths[1]))

noutdir = os.path.join(basedir, 'KDEF', 'out_norm')
if not os.path.exists(noutdir):
    os.makedirs(noutdir)

def feather(img):
    inner_kernel = np.ones((15,15), np.uint8);
    outer_kernel = np.ones((5,5), np.uint8);
    a_sm = cv2.erode(img[...,3], inner_kernel)
    a_bg = cv2.dilate(img[...,3], outer_kernel)
    d = cv2.distanceTransform(a_bg, cv2.DIST_L2, 5)
    d = d/d[(a_big-a_sm)>0].max()
    d[a_sm>0] = 1
    d = cv2.GaussianBlur(d,(5,5),0)
    img[...,3] = (d*255).round().astype(np.uint8)
    return img


for v in versions:
    for m in models:
        for em in emos:
            d = os.path.join(outdir, ''.join(v)+m+'_'+em)
            ims = glob(os.path.join(d, '*'))
            print('Histogram normalizing %s (%d images)...' % (d,len(ims)))
            od = os.path.join(noutdir, ''.join(v)+m+'_'+em)
            if not os.path.exists(od):
                os.mkdir(od)
            for imfn in ims:
                if os.path.exists(imfn):
                    img = cv2.imread(imfn, -1)
                    alpha = img[...,3]
                    rgb = img[...,:3]
                    cv2.GaussianBlur(rgb,(5,5),0)
                    img_yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
                    # equalize the histogram of the Y channel
                    #img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                    lum = img_yuv[alpha>0,0]
                    scale = 128. / lum.astype(np.float).mean()
                    img_yuv[...,0] = (img_yuv[...,0].astype(np.float) * scale).round().clip(0,255).astype(np.uint8)
                    # convert the YUV image back to RGB format
                    img_out = np.dstack((cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR), img[...,3]))
                    img_out = feather(img_out)
                    cv2.imwrite(os.path.join(od, os.path.basename(imfn)), img_out)









def plot_points():
    import numpy as np
    from matplotlib import pyplot as plt

    from PIL import Image
    import scipy
    impath = imgpaths[0]
    img = scipy.ndimage.imread(impath)[..., :3]
    points = fm.locator.face_points(impath)
    plt.imshow(img)
    plt.plot(points[:,0], points[:,1], '.')


