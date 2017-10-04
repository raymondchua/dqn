from __future__ import print_function, division

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

class DQN(nn.Module):
	def __init__(self, num_actions, use_bn=False):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)	
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

		if use_bn:
			self.use_bn = True
		else:
			self.use_bn = False

	def forward(self, x):
		if self.use_bn:
			x = F.relu(self.bn1(self.conv1(x)))
			x = F.relu(self.bn2(self.conv2(x)))
			x = F.relu(self.bn3(self.conv3(x)))
			x = F.relu(self.fc1(x.view(x.size(0), -1)))
			return self.fc2(x)

		else:
			x = F.relu(self.conv1(x))
			x = F.relu(self.conv2(x))
			x = F.relu(self.conv3(x))
			x = F.relu(self.fc1(x.view(x.size(0), -1)))
			return self.fc2(x.view(x.size(0), -1))
