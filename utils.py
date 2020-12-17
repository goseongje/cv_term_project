import os
import sys
import glob
import time
import torch
import torch.nn as nn
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

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)        

# it has been proved that this PSNR in torch is correct when batch size is 1
class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0).cuda())
        max_val = torch.tensor(max_val).float().cuda()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def forward(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10