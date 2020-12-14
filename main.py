from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import *
from dataset import ImageDataLoader

def main():
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

    train_dataset = ImageDataLoader(args.datapath, split="train")
    val_dataset = ImageDataLoader(args.datapath, split="val")
    test_dataset = ImageDataLoader(args.datapath, split="test")

    # 모델 선택
    if args.model == 'basic':
        model = basic()
    else:
        print('no model')