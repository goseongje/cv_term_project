import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import glob
import PIL.Image as Image
import numpy as np
import cv2
import skimage
from torch.utils.data import Dataset, DataLoader
from utils import imshow, load_filelist

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ImageDataLoader(Dataset):
    def __init__(self, path, split="train"):
        super(ImageDataLoader, self).__init__()

        self.path = path
        self.split = split

        self.flist_target = load_filelist(self.path, self.split)
        self.flist_guide = load_filelist(self.path, self.split)
        self.flist_gt = load_filelist(self.path, self.split)
        
        self.total = len(self.flist_target)     


    def __len__(self):
        return self.total


    def __getitem__(self, index):
        try:
            item = self.load_images(index)
        except:
            print('loading error: ' + self.load_name(index))
            item = self.load_images(0)

        return item
 

    def transform(self, img):
        transform_tar = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),        
            AddGaussianNoise(0., 0.01)])
        
        transform_ref = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            AddGaussianNoise(0., 0.03)])
        
        transform_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])        
        return transform_tar(img), transform_ref(img), transform_gt(img) 
        

    def load_images(self, index):    
        # flist_target = load_filelist(self.path, self.split) # gray
        # flist_ref = load_filelist(self.path, self.split) # reference
        # flist_gt = load_filelist(self.path, self.split) # ground truth
            
        common_image = Image.open(self.flist_target[index]).convert('RGB')
        target_image, ref_image, gt_image = self.transform(common_image)
        
        croph, cropw = 256, 512
        i, j, h, w = transforms.RandomCrop.get_params(common_image, output_size=(croph, cropw)) 
        target_image = F.crop(target_image, i, j, h, w) 
        ref_image = F.crop(ref_image, i, j, h, w)

        # list_img = []
        # list_img.append(target_image)
        # list_img.append(ref_image)
        # list_img.append(gt_image)
        # imshow(list_img)
        return target_image, ref_image, gt_image     

    def load_name(self, index):
        return os.path.basename(self.flist_gt[index])           