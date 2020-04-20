# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:49:40 2020

@author: 贺琦琦
"""

import os, random
from os.path import join as pjoin
from PIL import Image
import numpy as np
from dnnbrain.dnn.base import ip
from dnnbrain.dnn.models import AlexNet
from dnnbrain.dnn.algo import GuidedSaliencyImage
import matplotlib.pyplot as plt

# prepare parameters
dnn = AlexNet()
stim_main_path = '/nfs/e3/ImgDatabase/ImageNet_2017/ILSVRC2017_DET/ILSVRC/Data/DET/train/ILSVRC2013_train/'
out_sal_path = '/nfs/s2/userhome/heqiqi/workingdir/trials/saliency_libresult_fc3/'
out_pos_path = '/nfs/s2/userhome/heqiqi/workingdir/trials/extra_10kinds/'
ratio_std = 0.5
ratio_percent = 0.7
i = 0
m = 0
type_num = 20
pic_num = 1000

# generate random typelist
typedir = os.listdir(stim_main_path)
typelist = []
for pictype in typedir:
    tmp = len(os.listdir(pjoin(stim_main_path, pictype))) > pic_num
    if tmp == 1:
        typelist.append(1)
        typelist[m] = pictype
        m = m + 1
    if m == type_num:
        break

# generate random pic sample for every type
for ptype in typelist:
    stim_sub_path = pjoin(stim_main_path, ptype)
    sample = random.sample(os.listdir(stim_sub_path), pic_num)
    img_all = np.zeros((len(sample), 224, 224))
    # generate saliency image
    for picname in sample:
        pic_path = pjoin(stim_sub_path, picname)
        image = Image.open(pic_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        index = eval(picname.split('_')[1].split('.')[0])
        # define target channel
        mask_code = [23, 294, 596, 883, 996]
        img_allsa = np.zeros((len(mask_code), 224, 224))
        for num in range(5):
            transs = mask_code[num]
            guided = GuidedSaliencyImage(dnn)
            guided.set_layer('fc3', transs)
            img_out = np.abs(guided.backprop(image))
            img_grey = np.max(img_out, 0)
            # save img_grey
            img_allsa[num] = img_grey
        # compute mean and std
        img_avg = np.mean(img_allsa, axis=0)
        img_out = ip.to_pil(img_avg, True)
        trg = f'{ptype}_{index}_fc3_random_guidedsaliency.JPEG'
        if not os.path.exists(pjoin(out_sal_path, ptype)):
            os.mkdir(pjoin(out_sal_path, ptype))
        img_out.save(pjoin(out_sal_path, ptype, trg))
        mean = np.mean(img_avg)
        std = np.std(img_avg)
        img_pro = np.int64(img_avg >= mean + ratio_std*std)
        img_all[i] = img_pro
        i = i + 1
    # save out
    img_allpos = np.sum(img_all, axis=0)
    img_allpos = np.int64(img_allpos >= i*ratio_percent)
    if not os.path.exists(pjoin(out_pos_path, ptype)):
        os.mkdir(pjoin(out_pos_path, ptype))
    plt.imsave(pjoin(out_pos_path, ptype, f'{ptype}_pos_ran.jpg'), img_allpos)
