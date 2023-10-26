from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread,imwrite
from csbdeep.utils import download_and_extract_zip_file, plot_some, axes_dict
from csbdeep.io import save_training_data
from csbdeep.data import RawData, create_patches
from csbdeep.data.transform import anisotropic_distortions
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

raw_data = RawData.from_folder (
    #basepath    = 'D:/T/iso_tubulin/third_try/',
    basepath    = r"K:\brain\gxy\care\deep_learning\training\data/",
    source_dirs = ['axis_moved'],
    target_dir  = 'axis_moved',
    axes        = 'ZCYX',
)

psf= np.loadtxt(r'K:\brain\gxy\care\deep_learning\PSF\1.txt')

anisotropic_transform = anisotropic_distortions (
    subsample = 1,
    psf       = psf/np.sum(psf), # use the actual PSF here
    psf_axes  = 'YX',
    poisson_noise  = True,
)

#save internals
images =raw_data.generator()
images=anisotropic_transform.generator(images)

# for i,(x, target, axes, mask) in enumerate(images):
    
    # imwrite('D:/internals/out3.tif',x.astype(np.int16),'tif')
    

X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (1,1,64,64),
    n_patches_per_image = 1000,
    transforms          = [anisotropic_transform],
)

assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

z = axes_dict(XY_axes)['Z']
X = np.take(X,0,axis=z)
Y = np.take(Y,0,axis=z)
XY_axes = XY_axes.replace('Z','')

save_training_data('data/my_training_data.npz', X, Y, XY_axes)

for i in range(5):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1))
    plot_some(np.moveaxis(X[sl],1,-1),np.moveaxis(Y[sl],1,-1),title_list=[np.arange(sl.start,sl.stop)])
    plt.show()
None;