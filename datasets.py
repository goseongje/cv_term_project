import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class SceneFlowDataLoader(data.Dataset):
    def __init__(self, root: str, mode, training, loader=default_loader):
        self.mode = mode
        self.training = training

    dirs = root


cropW = 512 
cropH = 256
datasetDir = './data/cityscapes'

transform = transforms.Compose([
    transforms.RandomCrop(size=(cropH, cropW)),
    transforms.ToTensor()
])

trainset = torchvision.datasets.Cityscapes(root=datasetDir, split='train', mode='fine',
                                           target_type=['instance', 'color', 'polygon'],
                                           transform=transform)

"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
"""