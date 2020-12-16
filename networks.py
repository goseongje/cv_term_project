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
		self.layers_2n = self.get_layers(block, 32, 32, stride=1)	#2, 3
		"""
		self.layers_4n = self.get_layers(block, 32, 32, stride=1)	#4, 5
		self.layers_6n = self.get_layers(block, 32, 32, stride=1)	#6, 7
		self.layers_8n = self.get_layers(block, 32, 32, stride=1)	#8, 9
		self.layers_10n = self.get_layers(block, 32, 32, stride=1)	#10, 11
		self.layers_12n = self.get_layers(block, 32, 32, stride=1)	#12, 13
		self.layers_14n = self.get_layers(block, 32, 32, stride=1)	#14, 15
		self.layers_16n = self.get_layers(block, 32, 32, stride=1)	#16, 17
		"""
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
		"""
		x = self.layers_4n(x)
		x = self.layers_6n(x)
		x = self.layers_8n(x)
		x = self.layers_10n(x)
		x = self.layers_12n(x)
		x = self.layers_14n(x)                                        
		x = self.layers_16n(x)
		"""
		x = self.conv2(x)
		return x


class Attention(nn.Module):
	def __init__(self, num_layers):
		super().__init__()
		self.num_layers = num_layers
        # input image c x h x w : 1 x 256 x 512
        # 첫번째 레이어 kernel = 5 X 5, stride = 2
		self.attn1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1,
								stride=1, padding=1, bias=False)
		self.sig = nn.Sigmoid()
		self.attn2 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1, 
								stride=1, padding=1, bias=False)

	def forward(self, x):
		x = self.sig(self.attn1(x))
		x = self.sig(self.attn2(x))
		return x              


class ResidualBlock3d(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
		super(ResidualBlock3d, self).__init__()
		# for 3-D regulation
		self.cont1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, 
										stride=stride, padding=1, bias=False) #35, 36, 37, 38
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
							   stride=1, padding=1, bias=False)	#31, 28, 25, 22
		self.bn2 = nn.BatchNorm2d(out_channels)		
		self.stride = stride

		if down_sample:
			self.down_sample = IdentityPadding(in_channels, out_channels, stride)
		else:
			self.down_sample = None


	def forward(self, x):
		shortcut = x

		out = self.cont1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv1(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)
		return out


class Regulation3d(nn.Module):
	def __init__(self, num_layers, block):
		super().__init__()
		self.num_layers = num_layers

		self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, 
								stride=1, padding=1, bias=False) #21
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3,
								stride=1, padding=1, bias=False) #22
		self.bn2 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)	

		self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, 
								stride=2, padding=1, bias=False) #23, 26, 29, 32
		self.bn3 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)

		self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,
								stride=1, padding=1, bias=False) #24, 25, 27, 28, 30, 31, 33, 34
		self.bn4 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)

		self.layers_2n = self.get_layers(block, 32, 32, stride=2)	# 35, 31			
		self.layers_4n = self.get_layers(block, 32, 32, stride=2)	# 36, 28
		self.layers_6n = self.get_layers(block, 32, 32, stride=2)	# 37, 25
		self.layers_8n = self.get_layers(block, 32, 32, stride=2)	# 38, 22

		self.output = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, 
										stride=1, padding=1, bias=False) #39   

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
		x = self.conv2(x)		
		x = self.bn2(x)
		x = self.relu(x)
		
		for i in range(4):
			x = self.conv3(x)		
			x = self.bn3(x)
			x = self.relu(x)
			x = self.conv4(x)		
			x = self.bn4(x)
			x = self.relu(x)
			x = self.conv4(x)		
			x = self.bn4(x)
			x = self.relu(x)				
		
		x = self.layers_2n(x)
		x = self.layers_4n(x)
		x = self.layers_6n(x)
		x = self.layers_8n(x)

		x = self.output(x)

		return x 

def resnet1():
	block = ResidualBlock
	# total number of layers if 8n + 2. if n is 2 then the depth of network is 18.	
	model = ResNet(8, block, 1)
	return model

def attention():
	model = Attention(1)
	return model

def regulation3d():
	block = ResidualBlock3d
	model = Regulation3d(0, block)
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