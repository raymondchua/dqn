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

Experience = namedtuple('Experience', ('state', 'action', 'reward','next_state', 'td_error'))

class PrioritizedReplay(object):
	"""
	Memory index to begin from 1 so that we can use integer division by 2 for quick indexing. 
	"""
	def __init__(self, N):
		self.capacity = N 
		self.memory = []
		self.position = 0
		self.prioritySum = 0
		self.memory[0] = None

	def push(self, state, action, reward, next_state, td_error):
		"""
		Insert sample into priority replay. 
		At the same time, keep track of the total priority value in the replay memory.
		"""
		if(len(self.memory) < self.capacity):
			self.memory.append(None)
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.position = (self.position + 1) % self.capacity
			self.prioritySum += td_error

		else:
			temp = self.memory[self.position]
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.position = (self.position + 1) % self.capacity
			self.prioritySum -= temp.td_error
			self.prioritySum += td_error

	def sort(self):
		i = len(self.memory) // 2
		while(i > 0):
			self.percDown(i)
			i -= 1

	def percDown(self, i):
		while(i * 2) <= len(self.memory):
			temp_maxChild = self.maxChild(i)

			if self.memory[i] < self.memory[temp_maxChild]:
				tmp = self.memory[i]
				self.memory[i] = self.memory[temp_maxChild]
				self.memory[temp_maxChild] = tmp

			i = temp_maxChild

	def maxChild(self, i):
		if (i*2) + 1 > len(self.memory):
			return i * 2

		else:
			if self.memory[i*2].td_error > self.memory[(i*2)+1]:
				return i * 2

			else:
				return (i * 2) + 1


	def sample(self):
		"""
		Extract one random sample weighted by priority values from the replay memory.
		"""
		maxPriority = math.floor(self.prioritySum)
		randPriority = random.randint(1, maxPriority)

		for i in range(len(self.memory)):
			current = self.memory[i]
			if current.td_error <= randPriority:
				chosen_sample = current
				break

			randPriority -= current.td_error

		return chosen_sample



	def __len__(self):
		return len(self.memory)


