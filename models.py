import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from utils import show_model

class IdentityPadding(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super(IdentityPadding, self).__init__()
		
		self.pooling = nn.MaxPool2d(1, stride=stride)
		self.add_channels = out_channels - in_channels
    
	def forward(self, x):
		out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
		out = self.pooling(out)
		return out
	
	
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(ResidualBlock, self).__init__()
		# for ResNet
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, #2
							stride=1, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, #3
							stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.stride = stride		
		
		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None


	def forward(self, x):
		shortcut = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(x)
		out = self.bn1(out)		

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, num_layers, block, net_num):
		super(ResNet, self).__init__()
		self.num_layers = num_layers
		self.net_num = net_num
		if net_num == 1:
			self.stride = 2
		else:
			self.stride = 1

		if net_num == 4:
			self.out_channels = 1
		else:
			self.out_channels = 32

		# input image size : 1 x 256 x 512(c x h x w)
        # input layer kernel = 5 X 5, stride = 2
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, #1
							   stride=self.stride, padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)

		# feature map size = 16x128x256					
		self.layers_2n = self.get_layers(block, 32, 32, stride=1)	# 2 ~ 17

        # output layer kernel = 3 X 3, feature map size = 16x128x256
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=3, 
							   stride=1, padding=1, bias=False)		


	def get_layers(self, block, in_channels, out_channels, stride):
		if stride == 2:
			down_sample = True
		else:
			down_sample = False

		layers_list = nn.ModuleList(
			[block(in_channels, out_channels, stride, down_sample)])
			
		for _ in range(self.num_layers - 1):
			layers_list.append(block(out_channels, out_channels))

		return nn.Sequential(*layers_list)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.layers_2n(x)
		f = self.conv2(x)
		return f


class Attention(nn.Module):
	def __init__(self):
		super().__init__()		
        # input image c x h x w : 4 x 32 x 128 x 256
        # kernel = 1, stride = 1
		self.attn1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1,
								stride=1, padding=0, bias=False)
		self.sig = nn.Sigmoid()
		self.attn2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, 
								stride=1, padding=0, bias=False)
		self.sig = nn.Sigmoid()

	def forward(self, x):
		x = self.attn1(x)
		x = self.sig(x)
		x = self.attn2(x)
		x = self.sig(x)
		return x


class Regulation3d(nn.Module):
	#def __init__(self, num_layers, block):
	def __init__(self, num_layers):
		super().__init__()
		self.num_layers = num_layers

		self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, 
								stride=1, padding=1, bias=False) #21
		self.bn1 = nn.BatchNorm3d(32)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,
								stride=1, padding=1, bias=False) #22
		self.bn2 = nn.BatchNorm3d(32)
		self.relu = nn.ReLU(inplace=True)	

		self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, 
								stride=2, padding=1, bias=False) #23, 26, 29, 32
		self.conv3_1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, 
								stride=2, padding=1, bias=False) #23, 26, 29, 32								
		self.bn3 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)

		self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3,
								stride=1, padding=1, bias=False) #24, 25, 27, 28, 30, 31, 33, 34
		self.bn4 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)

		self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3,
								stride=1, padding=1, bias=False) #24, 25, 27, 28, 30, 31, 33, 34
		self.bn5 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)

		self.cont1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, 
										stride=2, padding=1, output_padding=1, bias=False) #35, 36, 37
		self.bn6 = nn.BatchNorm3d(64)
		self.relu = nn.ReLU(inplace=True)

		self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3,
							   stride=1, padding=1, bias=False)	#31, 28, 25, 22		

		self.cont2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, 
										stride=2, padding=1, output_padding=1, bias=False) #38
		self.bn7 = nn.BatchNorm3d(32)										

		self.output = nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=3, 
										stride=2, padding=1, output_padding=1, bias=False) #39   


	def forward(self, x):
		resdue = []
		x = self.conv1(x)		
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)	#22
		tmp = x
		x = self.bn2(x)
		x = self.relu(x)		
		
		# 31, 28, 25
		for i in range(4):
			if i == 0:
				x = self.conv3(x)	#23
			else:
				x = self.conv3_1(x)	#26, 29, 32		
			x = self.bn3(x)
			x = self.relu(x)	
			x = self.conv4(x)	#24, 27, 30, 33
			x = self.bn4(x)
			x = self.relu(x)					
			x = self.conv5(x)	#25, 28, 31, 34
			resdue.append(x)
			x = self.bn5(x)
			x = self.relu(x)		
		
		x = self.cont1(x)	#35
		x = self.bn6(x)
		x += resdue[2]	#35 + 31
		x = self.relu(x)

		x = self.cont1(x)	#36
		x = self.bn6(x)
		x += resdue[1]	#36 + 28
		x = self.relu(x)

		x = self.cont1(x)	#37
		x = self.bn6(x)
		x += resdue[0]	#37 + 25
		x = self.relu(x)

		x = self.cont2(x)	#38
		x = self.bn7(x)
		#x += tmp	#22		
		x = self.relu(x)

		x = self.output(x)
		out = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
		return out 

class WeightVolGen(nn.Module):
	def __init__(self):
		super(WeightVolGen, self).__init__()
		self.block = ResidualBlock		

		# resnet1 for Y(gray image)
		self.resnet1_y = ResNet(8, self.block, 1)
		# resnet1 for YR(ref image)
		self.resnet1_yr = ResNet(8, self.block, 1)
		# attention for Concatenated feature volume
		self.attention = Attention()
		# 3-D Regulation		
		self.regulation = Regulation3d(1)
	
	def forward(self, target_image, guide_image, gt_image):
		print('input target image size : {}'.format(target_image.shape))
		feature_map1 = self.resnet1_y(target_image)
		print('input guide image size : {}'.format(guide_image.shape))
		feature_map2 = self.resnet1_y(guide_image)				
		print('input stacked fmap size : {}'.format(torch.stack([feature_map1, feature_map2], dim=2).shape))
		attn_weighted_fvol = self.attention(torch.stack([feature_map1, feature_map2], dim=2))
		print('input attn_weighted_fvol size : {}'.format(attn_weighted_fvol.shape))
		weight_volume = self.regulation(attn_weighted_fvol)
		print("dddddddddddddd")
		print(weight_volume.shape)
		print(gt_image.shape)
		rough = torch.cat([weight_volume, gt_image], dim=2)

		print(rough.shape)		
		return weight_volume

class ColResJointLearning(nn.Module):
	def __init__(self):
		super(ColResJointLearning, self).__init__()

	def forward(self, weight_volume):

		return 0
		


def resnet1():
	block = ResidualBlock	
	model = ResNet(8, block, 1)
	return model

def attention():
	model = Attention()
	return model

def regulation3d():
	block = ResidualBlock3d
	model = Regulation3d(1, block)
	return model

def resnet2():
	block = ResidualBlock	
	model = ResNet(8, block, 2)
	return model

def resnet3():
	block = ResidualBlock	
	model = ResNet(8, block, 3)
	return model

def resnet4():
	block = ResidualBlock	
	model = ResNet(8, block, 4)
	return model
"""
net1 = resnet1()
summary(net1, (1, 256, 512), device="cpu")
net = attention()
summary(net, (32, 32, 128, 512), device="cpu")
"""