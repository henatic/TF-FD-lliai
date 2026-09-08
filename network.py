from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vgg19


def define_tsnet(name, num_class, cuda=True):
	if name == 'vgg19':
		net = VGG19_KD(num_classes=num_class)
		# Adjust the first convolutional layer or pooling for CIFAR's 32x32 input if necessary, 
        # or rely on standard torchvision definitions depending on your base setup.
	elif name == 'resnet20':
		net = resnet20(num_class=num_class)
	elif name == 'resnet110':
		net = resnet110(num_class=num_class)
	else:
		raise Exception('model name does not exist.')

	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	else:
		net = torch.nn.DataParallel(net)

	return net


class VGG19_KD(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG19_KD, self).__init__()
        # Load the base VGG19 architecture without pretrained ImageNet weights
        base_vgg = vgg19(pretrained=False)
        self.features = base_vgg.features
        
        # Rebuild classifier for CIFAR 32x32 -> 1x1x512 spatial reduction
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Manually chunk the sequential features into blocks matching ResNet's stages
        # pool1=idx 4, pool2=idx 9, pool3=idx 18, pool4=idx 27, pool5=idx 36
        stem = self.features[:5](x)
        rb1 = self.features[5:19](stem)
        rb2 = self.features[19:28](rb1)
        rb3 = self.features[28:37](rb2)
        
        feat = rb3.view(rb3.size(0), -1)
        out = self.classifier(feat)
        
        # Emulate the (pre_activation, post_activation) tuple expected by train_kd.py
        return (stem, stem), (rb1, rb1), (rb2, rb2), (rb3, rb3), feat, out

    def get_channel_num(self):
        # Match expected channel return signatures if using VID/AFD loss modes
        return [64, 128, 256, 512, 512]


class DynamicConv2d(nn.Conv2d):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DynamicConv2d, self).__init__(
            max_in_channels, max_out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        # The layer tracks the maximum possible size, but can use smaller slices.

    def forward(self, x, active_out_channels):
        # x.size(1) tells us how many channels the previous layer actually output
        active_in_channels = x.size(1)
        
        # Slice the weight tensor: [active_out, active_in, kernel_h, kernel_w]
        weight_slice = self.weight[:active_out_channels, :active_in_channels, :, :]
        
        # Slice the bias tensor if it exists
        bias_slice = self.bias[:active_out_channels] if self.bias is not None else None

        # Perform the convolution with the dynamically sliced weights
        return F.conv2d(x, weight_slice, bias_slice, self.stride, self.padding, self.dilation, self.groups)


class VGG_Supernet(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG_Supernet, self).__init__()
        
        # Define maximum search space dimensions (standard VGG19 widths)
        self.conv1 = DynamicConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = DynamicConv2d(64, 64, kernel_size=3, padding=1)
        # ... additional layers ...
        
        # Classifier needs to dynamically adapt to the final convolution's output
        self.classifier = nn.Linear(512, num_classes) # 512 is max possible

    def forward(self, x, channel_config):
        # channel_config is a list provided by the RL controller, e.g., [32, 48, ...]
        
        # Block 1
        x = self.conv1(x, active_out_channels=channel_config[0])
        x = F.relu(x)
        x = self.conv2(x, active_out_channels=channel_config[1])
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # ... route through remaining blocks ...
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        
        # Slice the final linear layer weights to match the active incoming features
        active_in_features = x.size(1)
        weight_slice = self.classifier.weight[:, :active_in_features]
        bias_slice = self.classifier.bias
        out = F.linear(x, weight_slice, bias_slice)
        
        return out


class resblock(nn.Module):
	def __init__(self, in_channels, out_channels, return_before_act):
		super(resblock, self).__init__()
		self.return_before_act = return_before_act
		self.downsample = (in_channels != out_channels)
		if self.downsample:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
			self.ds    = nn.Sequential(*[
							nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
							nn.BatchNorm2d(out_channels)
							])
		else:
			self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self.ds    = None
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu  = nn.ReLU()
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2   = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = x

		pout = self.conv1(x) # pout: pre out before activation
		pout = self.bn1(pout)
		pout = self.relu(pout)

		pout = self.conv2(pout)
		pout = self.bn2(pout)

		if self.downsample:
			residual = self.ds(x)

		pout += residual
		out  = self.relu(pout)

		if not self.return_before_act:
			return out
		else:
			return pout, out


class resnet20(nn.Module):
	def __init__(self, num_class):
		super(resnet20, self).__init__()
		self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1     = nn.BatchNorm2d(16)
		self.relu    = nn.ReLU()

		self.res1 = self.make_layer(resblock, 3, 16, 16)
		self.res2 = self.make_layer(resblock, 3, 16, 32)
		self.res3 = self.make_layer(resblock, 3, 32, 64)

		self.avgpool = nn.AvgPool2d(8)
		self.fc      = nn.Linear(64, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.num_class = num_class

	def make_layer(self, block, num, in_channels, out_channels): # num must >=2
		layers = [block(in_channels, out_channels, False)]
		for i in range(num-2):
			layers.append(block(out_channels, out_channels, False))
		layers.append(block(out_channels, out_channels, True))
		return nn.Sequential(*layers)

	def forward(self, x):
		pstem = self.conv1(x) # pstem: pre stem before activation
		pstem = self.bn1(pstem)
		stem  = self.relu(pstem)
		stem  = (pstem, stem)

		rb1 = self.res1(stem[1])
		rb2 = self.res2(rb1[1])
		rb3 = self.res3(rb2[1])

		feat = self.avgpool(rb3[1])
		feat = feat.view(feat.size(0), -1)
		out  = self.fc(feat)

		return stem, rb1, rb2, rb3, feat, out

	def get_channel_num(self):
		return [16, 16, 32, 64, 64, self.num_class]

	def get_chw_num(self):
		return [(16, 32, 32),
				(16, 32, 32),
				(32, 16, 16),
				(64, 8 , 8 ),
				(64,),
				(self.num_class,)]

class resnet110(nn.Module):
	def __init__(self, num_class):
		super(resnet110, self).__init__()
		self.conv1   = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1     = nn.BatchNorm2d(16)
		self.relu    = nn.ReLU()

		self.res1 = self.make_layer(resblock, 18, 16, 16)
		self.res2 = self.make_layer(resblock, 18, 16, 32)
		self.res3 = self.make_layer(resblock, 18, 32, 64)

		self.avgpool = nn.AvgPool2d(8)
		self.fc      = nn.Linear(64, num_class)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.num_class = num_class

	def make_layer(self, block, num, in_channels, out_channels):  # num must >=2
		layers = [block(in_channels, out_channels, False)]
		for i in range(num-2):
			layers.append(block(out_channels, out_channels, False))
		layers.append(block(out_channels, out_channels, True))
		return nn.Sequential(*layers)

	def forward(self, x):
		pstem = self.conv1(x) # pstem: pre stem before activation
		pstem = self.bn1(pstem)
		stem  = self.relu(pstem)
		stem  = (pstem, stem)

		rb1 = self.res1(stem[1])
		rb2 = self.res2(rb1[1])
		rb3 = self.res3(rb2[1])

		feat = self.avgpool(rb3[1])
		feat = feat.view(feat.size(0), -1)
		out  = self.fc(feat)

		return stem, rb1, rb2, rb3, feat, out

	def get_channel_num(self):
		return [16, 16, 32, 64, 64, self.num_class]

	def get_chw_num(self):
		return [(16, 32, 32),
				(16, 32, 32),
				(32, 16, 16),
				(64, 8 , 8 ),
				(64,),
				(self.num_class,)]


def define_paraphraser(in_channels_t, k, use_bn, cuda=True):
	net = paraphraser(in_channels_t, k, use_bn)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	else:
		net = torch.nn.DataParallel(net)

	return net


class paraphraser(nn.Module):
	def __init__(self, in_channels_t, k, use_bn=True):
		super(paraphraser, self).__init__()
		factor_channels = int(in_channels_t*k)
		self.encoder = nn.Sequential(*[
				nn.Conv2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_t, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])
		self.decoder = nn.Sequential(*[
				nn.ConvTranspose2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.ConvTranspose2d(factor_channels, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.ConvTranspose2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		out = self.decoder(z)
		return z, out


def define_translator(in_channels_s, in_channels_t, k, use_bn=True, cuda=True):
	net = translator(in_channels_s, in_channels_t, k, use_bn)
	if cuda:
		net = torch.nn.DataParallel(net).cuda()
	else:
		net = torch.nn.DataParallel(net)

	return net


class translator(nn.Module):
	def __init__(self, in_channels_s, in_channels_t, k, use_bn=True):
		super(translator, self).__init__()
		factor_channels = int(in_channels_t*k)
		self.encoder = nn.Sequential(*[
				nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		return z
