from __future__ import print_function, division

import cv2
import gym
from gym import wrappers

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image

from rank_based_prioritized_replay import RankBasedPrioritizedReplay, Experience
from dqn_model import DQN
from scheduler import Scheduler
import util


import time

Optimizer = namedtuple("Optimizer", ["type", "kwargs"])

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

#Hyperparameters for validation
NUM_GAMES = 30
MAX_FRAMES_PER_GAME = 520000

def ddqn_compute_y(batch_size=32, batch=None, model=None, target=None, gamma=0.99):
	"""
	Compute the Double Q learning error as based on the paper, 
	"Deep Reinforcement Learning with Double Q-learning" by Hado van Hasselt and
	Arthur Guez and David Silver. 
	Refer to equation 4 for the Double Q-learning error function.
	"""

	non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))) #to get a boolean value of 1 if not final 
	non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

	state_batch = Variable(torch.cat(batch.state)) #use cat to change data from tuple to tensor
	reward_batch = Variable(torch.cat(batch.reward)) 
	action_batch = Variable(torch.cat(batch.action))

	# #compute Q(s,a) based on the action taken
	state_action_values = model(state_batch).gather(1,action_batch)

	model_actions = model(non_final_next_states).data.max(1)[1].view(batch_size,1)
	model_action_batch = Variable(torch.cat([model_actions]), volatile=True)

	next_state_action_values = Variable(torch.zeros(batch_size)).type(Tensor)
	next_state_action_values[non_final_mask] = target(non_final_next_states).gather(1, model_action_batch)
	next_state_action_values.volatile = False

	y_output = reward_batch + (gamma * next_state_action_values)
	y_output = y_output.view(batch_size,1)
	
	# Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, y_output)

	return loss


def ddqn_rank_train(env, scheduler, optimizer_constructor, model_type, batch_size, rp_start, rp_size, 
	exp_frame, exp_initial, exp_final, inital_beta, gamma, target_update_steps, frames_per_epoch, 
	frames_per_state, output_directory, last_checkpoint):

	"""
	Implementation of the training algorithm for DDQN using Rank-based prioritization.
	Information with regards to the algorithm can be found in the paper, 
	"Prioritized Experience Replay" by Tom Schaul, John Quan, Ioannis Antonoglou and
	David Silver. Refer to section 3.3 in the paper for more info. 
	"""
	
	gym.undo_logger_setup()
	logging.basicConfig(filename='ddqn_training.log',level=logging.INFO)
	num_actions = env.action_space.n
	env.reset()
	
	print('No. of actions: ', num_actions)
	print(env.unwrapped.get_action_meanings())

	# initialize action value and target network with the same weights
	model = DQN(num_actions, use_bn=False)
	target = DQN(num_actions, use_bn=False)

	if use_cuda:
		model.cuda()
		target.cuda()

	frames_count = 1

	if last_checkpoint:
		model.load_state_dict(torch.load(last_checkpoint))
		print(last_checkpoint)
		print('weights loaded...')

		exp_replay = util.initialize_rank_replay_resume(env, rp_start, rp_size, frames_per_state, 
			model, target, gamma, batch_size)
		frames_count = get_index_from_checkpoint_path(last_checkpoint)

	else:
		exp_replay = util.initialize_rank_replay(env, rp_start, rp_size, frames_per_state, 
			model, target, gamma)

	target.load_state_dict(model.state_dict())

	temp = exp_replay.pop()

	print(temp.td_error)
	print(exp_replay.get_maxPriority())

	optimizer = optimizer_constructor.type(model.parameters(), lr=optimizer_constructor.kwargs['lr'],
		alpha=optimizer_constructor.kwargs['alpha'], eps=optimizer_constructor.kwargs['eps'] )

	episodes_count = 1
	frames_per_episode = 1
	epsiodes_durations = []
	rewards_per_episode = 0
	rewards_duration = []
	loss_per_epoch = []

	
	current_state, _, _, _ = util.play_game(env, frames_per_state)
	print('Starting training...')

	count = 0

	while True:

		epsilon=scheduler.anneal_linear(frames_count)
		choice = random.uniform(0,1)

		# epsilon greedy algorithm
		if choice <= epsilon:
			action = LongTensor([[random.randrange(num_actions)]])

		else:
			action = get_greedy_action(model, current_state)

		curr_obs, reward, done, _ = util.play_game(env, frames_per_state, action[0][0])

		rewards_per_episode += reward
		reward = Tensor([[reward]])

		current_state_ex = np.expand_dims(current_state, 0)
		curr_obs_ex = np.expand_dims(curr_obs, 0)
		action = action.unsqueeze(0)

		batch = Experience(current_state_ex, action, reward, curr_obs_ex, 0)

		#compute td-error for one sample
		td_error = ddqn_compute_y(batch_size=1, batch=batch, model=model, target=target, gamma=gamma).data.cpu().numpy()
		exp_replay.push(current_state, action, reward, curr_obs, td_error)

		current_state = curr_obs

		for j in range(batch_size):

			#Get a random sample
			obs_sample, obs_rank = exp_replay.sample()
			
			max_weight = exp_replay.get_max_weight(inital_beta)
			p_j = 1/obs_rank
			curr_weight = ((1/len(exp_replay))*(1/p_j))**inital_beta
			curr_weight = curr_weight/max_weight

			print(obs_rank)

		break
			
		# 	loss = ddqn_compute_y(obs_sample, batch_size, model, target, gamma)
		# 	optimizer.zero_grad()
		# 	loss.backward()

		# 	for param in model.parameters():
		# 		param.grad.data.clamp_(-1,1)

		# 	optimizer.step()
		# 	loss_per_epoch.append(loss.data.cpu().numpy())

	# 	frames_count+= 1
	# 	frames_per_episode+= frames_per_state

	# 	if done:
	# 		rewards_duration.append(rewards_per_episode)
	# 		rewards_per_episode = 0
	# 		frames_per_episode=1
	# 		episodes_count+=1
	# 		env.reset()
	# 		current_state, _, _, _ = util.play_game(env, frames_per_state)

	# 		if episodes_count % 100 == 0:
	# 			avg_episode_reward = sum(rewards_duration)/100.0
	# 			avg_reward_content = 'Episode from', episodes_count-99, ' to ', episodes_count, ' has an average of ', avg_episode_reward, ' reward and loss of ', sum(loss_per_epoch)
	# 			print(avg_reward_content)
	# 			logging.info(avg_reward_content)
	# 			rewards_duration = []
	# 			loss_per_epoch = []

	# 	# update weights of target network for every TARGET_UPDATE_FREQ steps
	# 	if frames_count % target_update_steps == 0:
	# 		target.load_state_dict(model.state_dict())
	# 		# print('weights updated at frame no. ', frames_count)


	# 	#Save weights every 250k frames
	# 	if frames_count % 250000 == 0:
	# 		torch.save(model.state_dict(), output_directory+model_type+'/weights_'+ str(frames_count)+'.pth')


	# 	#Print frame count for every 1000000 (one million) frames:
	# 	if frames_count % 1000000 == 0:
	# 		training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count, 'epsilon: ', epsilon
	# 		print(training_update)
	# 		logging.info(training_update)





