from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'


from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, IsotropicCARE

(X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data.npz', validation_split=0.1, verbose=True)
print('X : ' + str(X.shape))
print('Y : ' + str(Y.shape))

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=100, train_batch_size=4)
print(config)
vars(config)


model_name = '20230626_beadfullROI-2_desidelobe_gxy'


path_base = os.path.join('model/', model_name )
model = IsotropicCARE(config, 'my_model', basedir=path_base)

history = model.train(X,Y, validation_data=(X_val,Y_val))

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])