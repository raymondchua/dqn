from __future__ import print_function, division

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import time
import copy

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
		self.memory = {}
		self.sorted_memory = {}
		self.position = 1
		self.prioritySum = 0
		self.minPriority = None


	def push(self, state, action, reward, next_state, td_error):
		"""
		Insert sample into priority replay. 
		At the same time, keep track of the total priority value in the replay memory.
		"""
		if(len(self.memory) < self.capacity):
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			# self.priorityWeights[self.position] = td_error.data[0]
			# self.prioritySum += td_error.data[0]
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position = 1

			# if self.minPriority is None:
			# 	self.minPriority = td_error.data[0]
			# elif self.minPriority > td_error.data[0]:
			# 	self.minPriority = td_error.data[0]
			
		else:
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			# self.priorityWeights[self.position] = td_error.data[0]
			# self.prioritySum += td_error.data[0]
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position = 1
			# if self.minPriority > td_error.data[0]:
			# 	self.minPriority = td_error.data[0]

	def sort(self):
		self.sorted_memory = {}
		self.sorted_memory = copy.deepcopy(self.memory)
		i = len(self.sorted_memory) // 2
		while(i > 0):
			self.percDown(i)
			i -= 1

		self.updatePrioritySumAndMinPriority(self.sorted_memory)

	def percDown(self, i):
		while(i * 2) <= len(self.sorted_memory):
			temp_maxChild = self.maxChild(i)

			if (self.sorted_memory[i].td_error < self.sorted_memory[temp_maxChild].td_error).data.all():
				tmp = self.sorted_memory[i]
				self.sorted_memory[i] = self.sorted_memory[temp_maxChild]
				self.sorted_memory[temp_maxChild] = tmp

			i = temp_maxChild

	def maxChild(self, i):
		if ((i*2) + 1) > (len(self.sorted_memory)-1):
			return i * 2

		else:

			if (self.sorted_memory[i*2].td_error > self.sorted_memory[(i*2)+1].td_error).data.all():
				return i * 2

			else:
				return (i * 2) + 1

	def sample(self, batch_size):
		"""
		Extract one random sample weighted by priority values from the replay memory.
		"""
		samples_list = []
		rank_list = []
		priority_list = []
		segment_size = (len(self.sorted_memory)+1)//batch_size
		index = list(range(1,len(self.sorted_memory),segment_size))

		for i in index:
			if i + segment_size < len(self.sorted_memory):
				choice = random.randint(i, i+segment_size)
			else:
				choice = random.randint(i, len(self.sorted_memory)-1)
			samples_list.append(self.sorted_memory[choice])
			rank_list.append(choice)

			priorW = self.sorted_memory[choice].td_error/self.prioritySum

			if priorW < 1e-8:
				priorW = 1e-8

			priority_list.append(priorW)

		return samples_list, rank_list, priority_list

	def updatePrioritySumAndMinPriority(self, sorted_memory):

		self.prioritySum = 0
		self.minPriority = None

		for key, value in self.sorted_memory.items():
			self.prioritySum += value.td_error

			if self.minPriority == None:
				self.minPriority = value.td_error

			elif value.td_error < self.minPriority:
				self.minPriority = value.td_error


	def update(self, index, loss):
		"""
		update the samples new td values
		"""
		for i in range(len(index)):
			indexVal = index[i]
			curr_sample = self.sorted_memory[indexVal]
			self.prioritySum -= curr_sample.td_error.data[0]
			self.sorted_memory[indexVal] = Experience(curr_sample.state, curr_sample.action, curr_sample.reward, curr_sample.next_state, loss[i])
			self.priorityWeights[indexVal] = loss[i].data[0]
			self.prioritySum += loss[i].data[0]

			if self.minPriority > loss[i].data[0]:
				self.minPriority = loss[i].data[0]

			if self.minPriority == curr_sample.td_error.data[0]:
					self.minPriority = min(self.priorityWeights.values())

	def __len__(self):
		return len(self.memory)

	def get_minPriority(self):
		return self.minPriority

	def pop(self):
		return self.memory[1]

	def get_experience(self, index):
		return self.memory[index]


