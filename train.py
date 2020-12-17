import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torchsummary
import glob
import argparse
import models
from tqdm import tqdm
from torch.autograd import Variable
from dataset import ImageDataLoader
from utils import PSNR

parser = argparse.ArgumentParser(description='Weight volume generation by Resnet1')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--batch_size', default=256, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--epoch_num', type=int, default=10)
parser.add_argument('--num_worker', default=0, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
parser.add_argument('--model_name', type=str, default='weightvolgen')
parser.add_argument('--gpu_id', type=str ,default='0')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

epochs = args.epoch_num
batch_size = args.batch_size
gpu_id = args.gpu_id
lr = args.lr
seed = args.seed
num_worker = args.num_worker
out_dir = "./weight/checkpoint_model_{}_lr_{}_batchsize_{}".format(args.model_name, lr, batch_size)
save_dir = "./weight/checkpoint_model_{}_lr_{}_batchsize_{}".format(args.model_name, lr, batch_size)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
np.random.seed(seed)

# data load
#trainset = ImageDataLoader(path='./data/cityscapes/leftimg8bit', split='train_extra')
trainset = ImageDataLoader(path='./data/cityscapes/leftimg8bit', split='train')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=num_worker)

# define model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.WeightVolGen()
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0, momentum=0)
#psnr = PSNR(255.0).to(device)

for epochs in range(epochs):
    train_bar = tqdm(trainloader)

    for train_iter, items in enumerate(train_bar):
        model.train()

        # train 1
        target = Variable(items[0]).to(device)
        guide = Variable(items[1]).to(device)
        gt = Variable(items[2]).to(device)

        output = model(target, guide, gt)
        
        loss = 0
        mse_loss = criterion(output, gt)
        
        loss += mse_loss
        logs = [('l_mse', mse_loss.item())]

        model.backward(loss)
        
        # metrics
        psnr = psnr(gt, output)
        mae = (torch.sum(torch.abs(gt - target)) / torch.sum(gt)).float()
        logs.append(('psnr', psnr.item()))
        logs.append(('mae', mae.item()))

        # save model
        if train_iter % 1000 == 0:
            print('saving model...')
            save_name = '{}/'.format(save_dir) + args.model_name + '_' + 'epoch' + '_' + str(epoch) + '_' + 'mae' + str(mae) + '.pt'
            torch.save(model.state_dict(), save_name)
