from __future__ import print_function, division

import math
import random
import numpy as np
from collections import namedtuple
import time
import bisect
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
		self.memory = []
		self.position = 0


	def push(self, state, action, reward, next_state, td_error):
		"""
		Insert sample into priority replay. 
		At the same time, keep track of the total priority value in the replay memory.
		"""
		if(len(self.memory) < self.capacity):
			self.memory.append(None)
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.position = (self.position + 1) % (self.capacity)
			
		else:
		
			self.memory[self.position] = Experience(state, action, reward, next_state, td_error)
			self.position = (self.position + 1) % (self.capacity)
	
	def getKey(self, item):
		return item.td_error



	def sample(self, batch_size, sort=False):
		"""
		Extract sample from the replay memory.
		"""
		samples_list = []
		rank_list = []
		priority_list = []

		if sort:
			sorted(self.memory, key=self.getKey, reverse=True)

		if len(self.memory) < self.capacity:
			index = np.linspace(0, len(self.memory)-1, batch_size, endpoint=True, dtype=int)

		for i in range(len(index)):
			
			start = index[i]

			if i < len(index)-1:
				end = index[i+1]
			else:
				end = len(self.memory)

			choice = random.randint(start, end-1)

			curr_sample = self.memory[choice]

			#shift index by 1 since choice starts from 0
			segment_pvals = sum(range(start+1, end+1))
			prob_sample = (choice+1)/segment_pvals

			samples_list.append(curr_sample)
			priority_list.append(prob_sample)
			rank_list.append(choice)

		return samples_list, rank_list, priority_list

	def update(self, index, loss, new_sample):
		"""
		update the samples new td values
		"""

		new_td = new_sample.td_error
		insertNew = False

		for i in range(len(index)):
			indexVal = index[i]
			curr_loss = loss[i]

			if curr_loss < new_td and not insertNew:
				insertNew = True
				self.memory[indexVal] = new_sample

			elif i == len(index)-1 and not insertNew:
				insertNew = True
				self.memory[indexVal] = new_sample

			else:
				curr_sample = self.memory[indexVal]
				self.memory[indexVal] = Experience(curr_sample.state, curr_sample.action, curr_sample.reward, curr_sample.next_state, curr_loss)


	def __len__(self):
		return len(self.memory)

	def getReplayCapacity(self):
		return len(self.priorityQueue)

	def pop(self):
		return self.memory[1]

	def get_experience(self, index):
		return self.memory[index]
