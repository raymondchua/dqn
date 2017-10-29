from __future__ import print_function, division
import copy
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

def ddqn_compute_td_error(batch_size=32, state_batch=None, reward_batch=None, action_batch=None, next_state_batch=None,
	model=None, target=None, gamma=0.99):
	"""
	Compute the Double Q learning error as based on the paper, 
	"Deep Reinforcement Learning with Double Q-learning" by Hado van Hasselt and
	Arthur Guez and David Silver. 
	Refer to equation 4 for the Double Q-learning error function.
	"""
	#compute Q(s,a) based on the action taken
	state_action_values = model(state_batch).gather(1,action_batch)

	model_actions = model(next_state_batch).data.max(1)[1].view(batch_size,1)
	model_action_batch = Variable(torch.cat([model_actions]), volatile=True)

	next_state_action_values = Variable(torch.zeros(batch_size)).type(Tensor)
	next_state_action_values = target(next_state_batch).gather(1, model_action_batch)
	next_state_action_values.volatile = True

	y_output =  (gamma * next_state_action_values).add_(reward_batch) 
	
	state_action_values = state_action_values.squeeze()
	y_output = y_output.squeeze()

	loss =  (y_output - state_action_values).squeeze()

	return loss
	

def ddqn_compute_y(batch_size=32, state_batch=None, reward_batch=None, action_batch=None, next_state_batch=None, 
	model=None, target=None, gamma=0.99):
	"""
	Compute the Double Q learning error as based on the paper, 
	"Deep Reinforcement Learning with Double Q-learning" by Hado van Hasselt and
	Arthur Guez and David Silver. 
	Refer to equation 4 for the Double Q-learning error function.
	"""
	#compute Q(s,a) based on the action taken
	state_action_values = model(state_batch).gather(1,action_batch)

	model_actions = model(next_state_batch).data.max(1)[1].view(batch_size,1)
	model_action_batch = Variable(torch.cat([model_actions]), volatile=True)

	next_state_action_values = Variable(torch.zeros(batch_size)).type(Tensor)
	next_state_action_values = target(next_state_batch).gather(1, model_action_batch)
	next_state_action_values.volatile = False

	
	y_output =  (gamma * next_state_action_values).add_(reward_batch) 
	
	state_action_values = state_action_values.squeeze()
	y_output = y_output.squeeze()

	loss =  (y_output - state_action_values).squeeze()

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
	logging.basicConfig(filename='ddqn_rank_training.log',level=logging.INFO)
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
			action = util.get_greedy_action(model, current_state)

		curr_obs, reward, done, _ = util.play_game(env, frames_per_state, action[0][0])

		rewards_per_episode += reward
		reward = Tensor([[reward]])
		current_state_ex = Variable(current_state, volatile=True)
		curr_obs_ex = Variable(curr_obs, volatile=True)
		action_ex = Variable(action, volatile=True)
		reward_ex = Variable(reward, volatile=True)

		#compute td-error for one sample
		td_error = ddqn_compute_td_error(batch_size=1, state_batch=current_state_ex, reward_batch=reward_ex, action_batch=action_ex, 
			next_state_batch=curr_obs_ex, model=model, target=target, gamma=gamma)

		td_error = torch.abs(td_error)
		exp_replay.push(current_state_ex, action_ex, reward_ex, curr_obs_ex, td_error)
		current_state = curr_obs

		# compute y 
		if len(exp_replay) >= batch_size:
			# Get batch samples
			obs_samples, obs_ranks, obs_priorityVals = exp_replay.sample(batch_size)	
			obs_priorityTensor = torch.from_numpy(np.array(obs_priorityVals))
			p_batch = 1/ obs_priorityTensor
			w_batch = (1/len(exp_replay) * p_batch)**inital_beta
			max_weight = exp_replay.get_max_weight(inital_beta)
			params_grad = []

			for i in range(len(obs_samples)):
				sample = obs_samples[i]
				sample.state.volatile=False
				sample.next_state.volatile=False
				sample.reward.volatile=False
				sample.action.volatile=False
				loss = ddqn_compute_y(batch_size=1, state_batch=sample.state, reward_batch=sample.reward, action_batch=sample.action, 
					next_state_batch=sample.next_state, model=model, target=target, gamma=gamma)
				loss_abs = torch.abs(loss)
				exp_replay.update(obs_ranks[i], loss_abs)

				for param in model.parameters():
					if param.grad is not None:
						param.grad.data.zero_()

				loss.backward()

				#accumulate weight change
				if i == 0:
					for param in model.parameters():
						tmp = ((w_batch[i]/max_weight) * loss.data[0]) * param.grad.data
						params_grad.append(tmp)


				else:
					paramIndex = 0
					for param in model.parameters():
						tmp = ((w_batch[i]/max_weight) * loss.data[0]) * param.grad.data
						params_grad[paramIndex] = tmp + params_grad[paramIndex]
						paramIndex += 1
	
			# update weights
			paramIndex = 0
			for param in model.parameters():
				param.data += params_grad[paramIndex].mul(optimizer_constructor.kwargs['lr']).type(Tensor)
				paramIndex += 1
		
		frames_count+= 1
		frames_per_episode+= frames_per_state

		if done:
			# print('Game ends', rewards_per_episode)
			rewards_duration.append(rewards_per_episode)
			rewards_per_episode = 0
			frames_per_episode=1
			episodes_count+=1
			env.reset()
			current_state, _, _, _ = util.play_game(env, frames_per_state)

			if episodes_count % 100 == 0:
				avg_episode_reward = sum(rewards_duration)/100.0
				avg_reward_content = 'Episode from', episodes_count-99, ' to ', episodes_count, ' has an average of ', avg_episode_reward, ' reward and loss of ', sum(loss_per_epoch)
				print(avg_reward_content)
				logging.info(avg_reward_content)
				rewards_duration = []
				loss_per_epoch = []

		# update weights of target network for every TARGET_UPDATE_FREQ steps
		if frames_count % target_update_steps == 0:
			target.load_state_dict(model.state_dict())
			# print('weights updated at frame no. ', frames_count)


		#Save weights every 250k frames
		if frames_count % 250000 == 0:
			util.make_sure_path_exists(output_directory+model_type+'/')
			torch.save(model.state_dict(), 'rank_weights_'+ str(frames_count)+'.pth')


		#Print frame count and sort experience replay for every 1000000 (one million) frames:
		if frames_count % 1000000 == 0:
			training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count, 'epsilon: ', epsilon
			print(training_update)
			logging.info(training_update)
			exp_replay.sort()






