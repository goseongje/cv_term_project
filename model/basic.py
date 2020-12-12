from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
