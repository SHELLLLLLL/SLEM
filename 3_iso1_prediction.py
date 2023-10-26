from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import IsotropicCARE
import re
import imageio
import os
import time


def reverse_norm(im):
    max_ = np.max(im)
    min_ = np.min(im)
    # im = np.clip(im, min_, max_)
    im = (im - min_) / (max_ - min_ + 1e-10) * 65535
    return im.astype(np.uint16)


valid_lr_img_path = r'G:\gxy\brain\gxy\care\deep_learning\test/'
# model_name = '20230506_raw_laysom_650nm_Zpsf'
model_name = '20230626_beadfullROI-2_desidelobe_gxy'



def get_file_list(path, regx):
    file_list = os.listdir(path)
    file_list = [f for _, f in enumerate(file_list) if re.search(regx, f)]
    return file_list


axes = 'ZYX'
subsample = 1
print('image axes       =', axes)
print('subsample factor =', subsample)
path_base = os.path.join('model/', model_name )
model = IsotropicCARE(config=None, name='my_model', basedir=path_base)
valid_lr_imgs = get_file_list(path=valid_lr_img_path, regx='.*.tif')
for im_idx, im_file in enumerate(valid_lr_imgs):
    time_start = time.time()
    x = imageio.volread(os.path.join(valid_lr_img_path, im_file))
    restored = model.predict(x, axes, subsample)
    restored = reverse_norm(restored)
    save_dir = valid_lr_img_path + 'results'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save_tiff_imagej_compatible('results/%s20200823-0201_lr.tif' % model.name, restored, axes)
    save_tiff_imagej_compatible('%s/%s' % (save_dir, im_file), restored, axes)
    time_end = time.time()
    print('process:', im_file, 'finish_total_time:', time_end - time_start)
