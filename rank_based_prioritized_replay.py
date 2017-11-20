from __future__ import print_function, division

import math
import random
import numpy as np
from collections import namedtuple
import time

import torch

Experience = namedtuple('Experience', ('state', 'action', 'reward','next_state', 'td_error'))

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
		self.priorityWeights = []
		self.priorityQueue = []
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
			# if self.position == 1:
				#once memory is full, start index from mid-point to update the leaves
				# self.position = (self.position) % (self.capacity)
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			# self.priorityWeights[self.position] = td_error.data[0]
			# self.prioritySum += td_error.data[0]
			self.position = (self.position + 1) % (self.capacity)
			if self.position == 0:
				self.position = 1
			# if self.minPriority > td_error.data[0]:
			# 	self.minPriority = td_error.data[0]

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

	def getKey(self, item):
		return item.td_error.data[0]

	def build_new_replay(self): 
		self.priorityQueue = []
		self.prioritySum = 0

		for i in range(1, len(self.memory)):
			self.priorityQueue.append(self.memory[i])
			self.prioritySum += self.memory[i].td_error.data[0]

			if self.minPriority == None:
				self.minPriority = self.memory[i].td_error.data[0]

			elif self.memory[i].td_error.data[0] < self.minPriority:
				self.minPriority = self.memory[i].td_error.data[0]

	def build_new_pweights(self):
		self.priorityWeights = []

		for i in range(len(self.priorityQueue)):
			self.priorityWeights.append(self.priorityQueue[i].td_error.data[0])

		self.priorityWeights = np.array(self.priorityWeights)


	def sample(self, batch_size, iteration):
		"""
		Extract one random sample weighted by priority values from the replay memory.
		"""
		samples_list = []
		rank_list = []
		priority_list = []


		#get new replay when size of memory is less than capacity or for every 1000 frames
		if len(self.memory) < self.capacity:
			self.build_new_replay()
			sorted(self.priorityQueue, key=self.getKey)
			self.build_new_pweights()

		elif iteration %100 == 0:
			self.build_new_replay()
			sorted(self.priorityQueue, key=self.getKey)
			self.build_new_pweights()


		segment_size = len(self.priorityQueue)//batch_size
		index = list(range(0,len(self.priorityQueue),segment_size))


		for i in index:
			if i + segment_size < len(self.priorityQueue):
				choice = random.randint(i, i+segment_size)
				segment_total = np.sum(self.priorityWeights[i:i+segment_size])
			else:
				choice = random.randint(i, len(self.priorityQueue)-1)
				segment_total = np.sum(self.priorityWeights[i:len(self.priorityQueue)])

			samples_list.append(self.priorityQueue[choice])
			rank_list.append(choice)
			priorW = self.priorityQueue[choice].td_error.data[0]/segment_total

			if priorW < 1e-8:
				priorW = 1e-8

			priority_list.append(priorW)

		return samples_list, rank_list, priority_list

	def update(self, index, loss):
		"""
		update the samples new td values
		"""
		for i in range(len(index)):
			indexVal = index[i]
			curr_sample = self.priorityQueue[indexVal]
			self.prioritySum -= curr_sample.td_error.data[0]
			self.priorityQueue[indexVal] = Experience(curr_sample.state, curr_sample.action, curr_sample.reward, curr_sample.next_state, loss[i])
			self.prioritySum += loss[i].data[0]
			self.priorityWeights[i] = loss[i].data[0]

		self.minPriority = min(self.priorityWeights)

	def __len__(self):
		return len(self.memory)

	def getReplayCapacity(self):
		return len(self.priorityQueue)

	def pop(self):
		return self.memory[1]

	def get_experience(self, index):
		return self.memory[index]
