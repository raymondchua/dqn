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

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class RankBasedPrioritizedReplay(object):
	"""
	Implementation of Rank Based Prioritized Replay.

	Probability of sampling transition i is defined as :
		p_i = 1 / rank(i)
	where rank(i) is the rank of transition i when the replay memory is sorted according to td-error.
	Refer to the paper, "Priotitized Experience Replay" section 3.3 for more info.
	
	Data structure used to approximate the sorted priority queue is a binary heap.  
	Memory index to begin from 1 so that we can use integer division by 2 for quick indexing. 
	"""
	def __init__(self, N):
		self.capacity = N
		self.memory = []
		self.position = 1
		self.memory.append(None)
		self.priorityWeights = torch.zeros(N).type(Tensor)


	def push(self, state, action, reward, next_state, td_error):
		"""
		Insert sample into priority replay. 
		At the same time, keep track of the total priority value in the replay memory.
		"""
		if(len(self.memory) < self.capacity):
			self.memory.append(None)
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.priorityWeights[self.position] = td_error.data[0]
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position += 1
			
		else:
			temp = self.memory[self.position]
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.priorityWeights[self.position] = td_error.data[0]
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position += 1

	def sort(self):
		i = len(self.memory) // 2
		while(i > 0):
			self.percDown(i)
			i -= 1

	def percDown(self, i):
		while(i * 2) <= len(self.memory):
			temp_maxChild = self.maxChild(i)

			if (self.memory[i].td_error < self.memory[temp_maxChild].td_error).data.all():
				tmp = self.memory[i]
				self.memory[i] = self.memory[temp_maxChild]
				self.memory[temp_maxChild] = tmp

			i = temp_maxChild

	def maxChild(self, i):
		if ((i*2) + 1) > (len(self.memory)-1):
			return i * 2

		else:

			if (self.memory[i*2].td_error > self.memory[(i*2)+1].td_error).data.all():
				return i * 2

			else:
				return (i * 2) + 1


	def normalize_weights(self):
		self.priorityWeights = torch.div(self.priorityWeights,torch.sum(self.priorityWeights))




	def sample(self, batch_size):
		"""
		Extract one random sample weighted by priority values from the replay memory.
		"""
		samples_list = []
		rank_list = []
		priority_list = []
		self.normalize_weights()
		samples = torch.utils.data.sampler.WeightedRandomSampler(self.priorityWeights.tolist(), batch_size, replacement=False)
		for sample in samples:
			samples_list.append(self.memory[sample])
			rank_list.append(sample)
			priority_list.append(self.priorityWeights[sample])

		return samples_list, rank_list, priority_list

	def update(self, index, loss):
		"""
		update the samples new td values
		"""
		curr_sample = self.memory[index]
		self.memory[index] = Experience(curr_sample.state, curr_sample.action, curr_sample.reward, curr_sample.next_state, loss.data[0])
		self.priorityWeights[index] = loss.data[0]


	def get_max_weight(self, beta):
		temp = self.priorityWeights[self.priorityWeights != 0] / torch.sum(self.priorityWeights)
		return ((1/(len(self.memory))) * 1/(torch.min(temp))) ** beta

	def __len__(self):
		return len(self.memory)

	def get_minPriority(self):
		return self.minPriority

	def pop(self):
		return self.memory[1]

	def get_experience(self, index):
		return self.memory[index]

	def get_maxPriority(self):
		return self.maxPriority


