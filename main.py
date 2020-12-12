from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import *

parser = argparse.ArgumentParser(description='Sensor Fusion')
parser.add_argument('--model', default='basic',
                    help='select model')
parser.add_argument('--datapath', default='./data/cityscapes',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=0,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= './model/pretrained_cityscapes.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./model',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Data Processing
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

# 모델 선택
if args.model == 'basic':
    model = basic()
else:
    print('no model')

def main():
    parser = argparse.ArgumentParser(description='Training of Network using training data')
    parser.add_argument('--gpu', type=str, default="0,1", help='gpu id')
    args = parser.parse_args()    