import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from PIL import Image, ImageDraw
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils.lsun_loader as lsun_loader
import utils.svhn_loader as svhn
from utils.tinyimages_80mn_loader import RandomImages
from utils.imagenet_rc_loader import ImageNet

import pathlib

cifar10_path = '../data/cifarpy'
cifar100_path = '../data/cifar-100-python'
svhn_path = '../data/svhn/'
lsun_c_path = '../data/LSUN_C'
lsun_r_path = '../data/LSUN_resize'
isun_path = '../data/iSUN'
dtd_path = '../data/dtd/images'
places_path = '../data/places365/'
tinyimages_300k_path = '../data/300K_random_images.npy'
svhn_path = '../data/svhn'


def load_tinyimages_300k():
    print('loading TinyImages-300k')
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    t = trn.Compose([trn.ToTensor(),
                     trn.ToPILImage(),
                     trn.ToTensor(),
                     trn.Normalize(mean, std)])

    data = RandomImages(root=tinyimages_300k_path, transform=t)

    return data


def load_dataset(dataset):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if dataset == 'lsun_c':
        print('loading LSUN_C')
        out_data = dset.ImageFolder(root=lsun_c_path,
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std),
                                                        trn.RandomCrop(32, padding=4)]))#随机裁剪成(32,32)

    """
            out_data = dset.ImageFolder(root=lsun_c_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std),
                                                           trn.RandomCrop(32, padding=4)]))#随机裁剪成(32,32)
    """
    if dataset == 'lsun_r':
        print('loading LSUN_R')
        out_data = dset.ImageFolder(root=lsun_r_path,
                                    transform=None)  # transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))

    if dataset == 'isun':
        print('loading iSUN')
        out_data = dset.ImageFolder(root=isun_path,
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    if dataset == 'dtd':
        print('loading DTD')
        out_data = dset.ImageFolder(root=dtd_path,
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
    if dataset == 'places':
        print('loading Places365')
        out_data = dset.ImageFolder(root=places_path,
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))

    return out_data

"""看经过处理的图片的数据
#data=load_tinyimages_300k()
data=load_dataset('lsun_c')
max=-1
min=2
for d,t in data:
    #print(d[0].shape)
    if max<torch.max(d[2]):
        max=torch.max(d[2])
    if min>torch.min(d[2]):
        min=torch.min(d[2])
print(max)#lsun_r/lsun_c/dtd/places/isun/tinyimages:2.1256    
print(min)#       -1.9889
          #每张经过处理的图片都是tensor[3,32,32]
"""

"""
#图片的数据
data=load_dataset('lsun_r')  #ToTensor（） 把shape=(H x W x C) 的像素值为 [0, 255] 的 PIL.Image 和 numpy.ndarray 转换成shape=(C x H x W)的像素值范围为[0.0, 1.0]的 torch.FloatTensor
#print(data.class_to_idx)    #{'test': 0} 
print(data[1][0].size) 
print(data[1][0].getpixel((25,10)))         #第一维表示第几个数据、第二维：transform前后都是[0]：数据、[1]:标签0，只不过transform前是一个图片数据，transform后是一个tensor
                                            #print(data[1][0].getpixel((30,32)))   位置为(30,32)的数据的像素点：(25, 44, 102)
"""

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
tensordata=torch.randint(0,256,(3,2,2))#randint(0,1)只会生成0
#tensordata=torch.ones(3,2,2)
tensordata=tensordata/255

data = trn.Normalize(mean, std)(tensordata)
data=trn.Normalize(list(-np.array(mean)/np.array(std)), list(1/np.array(std)))(data)
data=data*255
print(data)
#print(tensordata)
#print(torch.min(data))
#unloader = transforms.ToPILImage()

"""
screen -S openauc
cd CIFAR
conda activate OpenAUC
bash run.sh
"""