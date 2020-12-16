import os
import sys
import glob
import time
import torch
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def imshow(img):
    for i in range(len(img)):
        tmp_img = img[i] / 2 + 0.5     # unnormalize
        npimg = tmp_img.numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_filelist(path, split):    
    filelist = []
    # split = 'train'
    # image_dir = os.path.join('./data/cityscapes/leftimg8bit', split)
    image_dir = os.path.join(path, split)
    for city in os.listdir(image_dir):
        img_dir = os.path.join(image_dir, city)
        for file_name in os.listdir(img_dir):
            filelist.append(os.path.join(img_dir, file_name))    
    filelist.sort()
    return filelist

def show_model(net):
    for idx, m in enumerate(net.modules()):
        print(idx, '->', m)