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
		self.prioritySum = 0
		self.memory.append(None)
		self.minPriority = None
		self.maxPriority = None

	def push(self, state, action, reward, next_state, td_error):
		"""
		Insert sample into priority replay. 
		At the same time, keep track of the total priority value in the replay memory.
		"""
		if(len(self.memory) < self.capacity):
			self.memory.append(None)
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position += 1
			self.prioritySum += td_error

		else:
			temp = self.memory[self.position]
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position += 1
			self.prioritySum -= temp.td_error
			self.prioritySum += td_error

		if len(self.memory) == 2:
			self.minPriority = td_error
			self.maxPriority = td_error

		elif (td_error < self.minPriority).data.all():
			self.minPriority = td_error

		elif (td_error > self.maxPriority).data.all():
			self.maxPriority = td_error


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


	def sample(self):
		"""
		Extract one random sample weighted by priority values from the replay memory.
		"""
		maxPriority = torch.floor(self.prioritySum).data.type(Tensor)
		randNum = torch.rand(1).type(Tensor)
		randPriority = randNum * maxPriority
		
		for i in range(1, len(self.memory)):
			rank = i
			current = self.memory[i]
			td_error_float = current.td_error.data.type(Tensor)
		
			if (randPriority <= td_error_float).all() :
				chosen_sample = current
				chosen_sample_index = i 
				break

			randPriority -= current.td_error.data.type(Tensor)
		

		#swap selected sample with the last sample in the array
		self.memory[i] = self.memory[len(self.memory)-1]
		self.memory[len(self.memory)-1] = chosen_sample
		self.prioritySum -= chosen_sample.td_error

		return chosen_sample, rank

	def update(self, state, action, reward, next_state, new_td_error):
		"""
		Similar to push function, except that the updated sample is added to the queue as last element.
		At the same time, keep track of the total priority value in the replay memory.
		"""
		self.memory[len(self.memory)-1] = Experience(state, action, reward, next_state, new_td_error)
		self.prioritySum += new_td_error

		if len(self.memory) == 2:
			self.minPriority = td_error
			self.maxPriority = td_error

		elif (new_td_error < self.minPriority).data.all():
			self.minPriority = new_td_error

		elif (new_td_error > self.maxPriority).data.all():
			self.maxPriority = new_td_error

	def get_max_weight(self, beta):
		return ((1/(len(self.memory))) * 1/(self.minPriority/self.prioritySum)) ** beta

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


