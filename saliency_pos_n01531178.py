# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 23:16:18 2020

@author: 贺琦琦
"""

import os
from os.path import join as pjoin
from PIL import Image
import numpy as np
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.algo import GuidedSaliencyImage
import matplotlib.pyplot as plt

# prepare parameters
dnn = AlexNet()
stim_main_path = '/nfs/s2/userhome/heqiqi/workingdir/trials/stims/n01531178/'
out_sal_path = '/nfs/s2/userhome/heqiqi/workingdir/trials/saliency_libresult_fc3/n01531178/'
out_allse_path = '/nfs/s2/userhome/heqiqi/workingdir/trials/extra_10kinds/n01531178/'
ratio = 0.5
i = 0

# generate stim.csv if you want to compute activation

# generate saliency image
img_all = np.zeros((len(os.listdir(stim_main_path)), 224, 224))
for picname in os.listdir(stim_main_path):
    pic_path = pjoin(stim_main_path, picname)
    image = Image.open(pic_path)
    index = eval(picname.split('_')[1].split('.')[0])
    # define target channel
    mask_code = [23, 294, 596, 883, 996]
    img_allsa = np.zeros((len(mask_code), 224, 224))
    for num in range(5):
        transs = mask_code[num]
        guided = GuidedSaliencyImage(dnn)
        guided.set_layer('fc3', mask_code[num])
        img_out = np.abs(guided.backprop(image))
        img_grey = np.max(img_out, 0)
        # save img_grey
        img_allsa[num] = img_grey
        img_out = ip.to_pil(img_grey, True)
        trg = f'n01531178_{index}_fc3_{transs}_guidedsaliency.JPEG'
        img_out.save(pjoin(out_sal_path, trg))
    # compute mean and std
    img_avg = np.mean(img_allsa, axis=0)
    mean = np.mean(img_avg)
    std = np.std(img_avg)
    img_pro = np.int64(img_avg >= mean + ratio*std)
    img_all[i] = img_pro
    i = i + 1
    # save out
img_allse = np.sum(img_all, axis=0)
img_allse = np.int64(img_allse >= i*0.7)
plt.imsave(pjoin(out_allse_path, 'n01531178_pos20.jpg'), img_allse)
