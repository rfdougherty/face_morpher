#!/usr/bin/env python
import os.path
import sys
from time import strftime
from PIL import Image
from glob import glob

row_size = 7
margin = 3

imdir = '/Users/bobd/git/face_morpher/KDEF/raw/'
dirs = glob(os.path.join(imdir, '*'))


def generate_montage(filenames, output_fn):
    images = [Image.open(filename) for filename in filenames]

    width = max(image.size[0] + margin for image in images)*row_size
    height = sum(image.size[1] + margin for image in images)
    montage = Image.new(mode='RGB', size=(width, height), color=(0,0,0,0))

    max_x = 0
    max_y = 0
    offset_x = 0
    offset_y = 0
    for i,image in enumerate(images):
        montage.paste(image, (offset_x, offset_y))

        max_x = max(max_x, offset_x + image.size[0])
        max_y = max(max_y, offset_y + image.size[1])

        if i % row_size == row_size-1:
            offset_y = max_y + margin
            offset_x = 0
        else:
            offset_x += margin + image.size[0]

    montage = montage.crop((0, 0, max_x, max_y))
    montage.save(output_fn)

emos = ['NE','HA','SA','AN','AF','DI','SU']
models = [s+'%02d'%m for m in range(1, 36) for s in ['M','F']]

for m in models:
    all_ims = []
    print('  %s...' % m)
    for s in ['A','B']:
        for e in emos:
            fname = '%s%s%sS.JPG' % (s,m,e)
            all_ims.append(os.path.join(imdir,s+m,fname))
    generate_montage(all_ims, '/tmp/'+m+'.jpg')


for s in ['A','B']:
  for e in emos:
    all_ims = []
    for d in dirs:
        print('  %s...' % os.path.basename(d))
        ims = glob(os.path.join(d, s+'???'+e+'S.JPG'))
        for f in ims:
            all_ims.append(f)

    generate_montage(all_ims, '/tmp/'+s+e+'.jpg')

