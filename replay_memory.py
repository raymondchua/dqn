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

Experience = namedtuple('Experience', ('state', 'action', 'reward','next_state'))

class ExpReplay(object):
	def __init__(self, N):
		self.capacity = N 
		self.memory = []
		self.position = 0

	def push(self, state, action, reward, next_state):
		if(len(self.memory) < self.capacity):
			self.memory.append(None)
		self.memory[self.position] = Experience(state, action, reward, next_state)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)



