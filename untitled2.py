#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:29:52 2020

@author: shixiaoxuan
"""

import os.path

import random

import numpy as np
import cv2
import csv

filepathtotal = '/nfs/e3/ImgDatabase/ImageNet_2017/ILSVRC2017_DET/ILSVRC/Data/DET/train/ILSVRC2013_train'
filesavetotal = '/nfs/s2/userhome/shixiaoxuan/workingdir/ackd/stimulate'
pathdirtotal = os.listdir(filepathtotal)
m = 3
newdir = random.sample(pathdirtotal,m)
for file in range(0,m):
    filepath=os.path.join(filepathtotal,newdir[file])
#    print(filepath)
    pathdir = os.listdir(filepath)
    n=3
    newlist = random.sample(pathdir,n)
    filesave = os.path.join(filesavetotal,newdir[file])
    os.mkdir(filesave)
    name='path='+filesave
    title='title='+newdir[file]
    information=np.array(['type=image',name,title,'data=stimID'])
    for i in range(0,n):
        pathjpeg = os.path.join(filepath,newlist[i])
        img=cv2.imread(pathjpeg)
        crop_size= (224,224)
        img_new=cv2.resize(img, crop_size, interpolation=cv2.INTER_CUBIC)
        img_newname = 'new'+newlist[i]
        newpath = filesave + '/' + img_newname 
        print(newpath)
        cv2.imwrite(newpath,img_new)
        information=np.append(information,[img_newname])

    
    print(information)
    stimfilename = '%s/%s_stimulate.stim.csv' %(filesavetotal,newdir[file])
    print(stimfilename)
    os.mknod(stimfilename)
    fp= open(stimfilename,'w')
    csv_writer = csv.writer(fp)
    
    for a in range(0,n+4):
        fip=np.array(information[a])
        csv_writer.writerow([fip])
    fp.close()
    a=0
    i=0
    
 
    
    
    
    
    
    
    
'''
    csv_writer = csv.writer(fp)
    
    for a in range(0,n+3):
        fip=np.array(information[a])
        csv_writer.writerow(fip)
         writer =csv.writer(fp)
    np.savetxt(stimfilename,information,delimiter=',')


'''

    
    
    
   
    
    
    
    
    
                
        
'''
        count=0
for i in pathdir:
    count=count + 1

print(count)
'''