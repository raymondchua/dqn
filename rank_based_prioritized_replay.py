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



	def sample(self, batch_size, alpha, sort=False):
		"""
		Extract sample from the replay memory.
		"""
		samples_list = []
		rank_list = []
		priority_list = []

		segment_range = list(range(1, len(self.memory)))
		segment_pvals = [1/x for x in segment_range]
		segment_pvals_sum = sum(segment_pvals)

		
		if sort:
			sorted(self.memory, key=self.getKey)

		# if len(self.memory) < self.capacity:
		# 	index = np.linspace(0, len(self.memory)-1, batch_size, endpoint=True, dtype=int)

		for i in range(batch_size):
			
			# start = index[i]

			# if i < len(index)-1:
			# 	end = index[i+1]
			# else:
			# 	end = len(self.memory)

			# choice = random.randint(start, end-1)

			# curr_sample = self.memory[choice]

			choice = i*-1
			curr_sample = self.memory[choice]
			p_i = math.pow(1/(i+1), alpha)
			p_k = math.pow(segment_pvals_sum, alpha)
			prob_sample = p_i/p_k

			samples_list.append(curr_sample)
			priority_list.append(prob_sample)
			# rank_list.append(choice)

		return samples_list, priority_list

	def update(self, old_samples, loss, new_sample):
		"""
		update the samples new td values
		"""
		for i in range(len(old_samples)):
			curr_sample = old_samples[i]
			curr_loss = loss[i]
			# print(len(self.memory))
			# print(self.position)
			# self.push(curr_sample.state, curr_sample.action, curr_sample.reward, curr_sample.next_state, curr_loss)
			choice = i * -1
			self.memory[choice] = Experience(curr_sample.state, curr_sample.action, curr_sample.reward, curr_sample.next_state, curr_loss)

		#insert the new sample
		self.push(new_sample.state, new_sample.action, new_sample.reward, new_sample.next_state, new_sample.td_error)

	def __len__(self):
		return len(self.memory)

	def getReplayCapacity(self):
		return len(self.priorityQueue)

	def pop(self):
		return self.memory[1]

	def get_experience(self, index):
		return self.memory[index]
