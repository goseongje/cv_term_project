import torch.nn as nn
import torch.nn.functional as F

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
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
			                   stride=stride, padding=1, bias=False) 
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
			                   stride=1, padding=1, bias=False) 
		self.bn2 = nn.BatchNorm2d(out_channels)
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

		out = self.conv2(out)
		out = self.bn2(out)

		if self.down_sample is not None:
			shortcut = self.down_sample(x)

		out += shortcut
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, num_layers, block):
		super(ResNet, self).__init__()
		self.num_layers = num_layers
        # input image c x h x w : 1 x 256 x 512
        # 첫번째 레이어 kernel = 5 X 5, stride = 2
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, 
							   stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)
		
		# feature map size = 32x32x16
		self.layers_2n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 16x16x32
		self.layers_4n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 8x8x64
		self.layers_6n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 8x8x64
		self.layers_8n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 8x8x64
		self.layers_10n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 8x8x64
		self.layers_12n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 8x8x64
		self.layers_14n = self.get_layers(block, 16, 16, stride=1)
		# feature map size = 8x8x64
		self.layers_16n = self.get_layers(block, 16, 16, stride=1)                                        

        # 마지막 레이어 kernel = 3 X 3
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, 
							   stride=1, padding=1, bias=False)
		
		# output layers
		#self.avg_pool = nn.AvgPool2d(8, stride=1)
		#self.fc_out = nn.Linear(64, num_classes)
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', 
					                    nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
        """

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
		x = self.layers_4n(x)
		x = self.layers_6n(x)
		x = self.layers_8n(x)
		x = self.layers_10n(x)
		x = self.layers_12n(x)
		x = self.layers_14n(x)
		x = self.layers_16n(x)                                        

		x = self.conv2(x)
		return x


class Attention(nn.Module):
	def __init__(self, num_layers):
		super().__init__()
		self.num_layers = num_layers
        # input image c x h x w : 1 x 256 x 512
        # 첫번째 레이어 kernel = 5 X 5, stride = 2
		self.attn1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=1)
		self.sig = nn.Sigmoid()
        self.attn2 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.sig(self.attn1(x))
        x = self.sig(self.attn2(x))
        return x              


class regulation3d(nn.Module):
	def __init__(self, num_layers):
		super().__init__()
		self.num_layers = num_layers

        self.reg1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3) #21
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)		
        self.reg2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3) #22
        self.reg3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2)	#23
        self.reg4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3) #24, 25
        self.reg5 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, stride=2) #35, 36, 37
		self.reg6 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=2) #38
		self.reg7 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, stride=2) #39   
    def forward(self, x)
		x = self.reg1(x)
		x = self.bn1(x)
		x = self.relu(x)3
        return x 

def resnet():
	block = ResidualBlock
	# total number of layers if 8n + 2. if n is 2 then the depth of network is 18.
	model = ResNet(2, block) 
	return model

def attention():
    model = Attention(1)
    return model


net = resnet()