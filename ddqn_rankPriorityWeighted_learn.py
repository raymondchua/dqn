from __future__ import print_function, division
import copy
import cv2
import gym
from gym import wrappers

import math
import random
import numpy as np
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


from rank_based_prioritized_replay import RankBasedPrioritizedReplay, Experience
from dqn_model import DQN
import util
from WeightedLoss import Weighted_Loss


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
	

def ddqn_compute_y(batch, batch_size, model, target, gamma, weights, loss):
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

	weights_var = Variable(weights)

	#compute Q(s,a) based on the action taken
	state_action_values = model(state_batch).gather(1,action_batch)

	model_actions = model(non_final_next_states).data.max(1)[1].view(batch_size,1)
	model_action_batch = Variable(torch.cat([model_actions]), volatile=True)

	next_state_action_values = Variable(torch.zeros(batch_size)).type(Tensor)
	next_state_action_values[non_final_mask] = target(non_final_next_states).gather(1, model_action_batch)
	next_state_action_values.volatile = False

	y_output =  (next_state_action_values*gamma) + reward_batch.squeeze()

	new_weights = state_action_values.squeeze() -  y_output

	y_output = y_output.view(batch_size,1)
	
	lossVal = loss(state_action_values, y_output, weights_var)

	return lossVal, new_weights

	

def ddqn_rankWeight_train(env, exploreScheduler, betaScheduler, optimizer_constructor, model_type, batch_size, rp_start, rp_size, 
	exp_frame, exp_initial, exp_final, prob_alpha, gamma, target_update_steps, frames_per_epoch, 
	frames_per_state, output_directory, last_checkpoint, max_frames):

	"""
	Implementation of the training algorithm for DDQN using Rank-based prioritization.
	Information with regards to the algorithm can be found in the paper, 
	"Prioritized Experience Replay" by Tom Schaul, John Quan, Ioannis Antonoglou and
	David Silver. Refer to section 3.3 in the paper for more info. 
	"""
	
	gym.undo_logger_setup()
	logging.basicConfig(filename='ddqn_rank_weightedLoss_training.log',level=logging.INFO)
	num_actions = env.action_space.n
	env.reset()
	
	print('No. of actions: ', num_actions)
	print(env.unwrapped.get_action_meanings())

	# initialize action value and target network with the same weights
	model = DQN(num_actions)
	target = DQN(num_actions)

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
			model, target, gamma, prob_alpha)

	target.load_state_dict(model.state_dict())

	optimizer = optimizer_constructor.type(model.parameters(), lr=optimizer_constructor.kwargs['lr'],
		alpha=optimizer_constructor.kwargs['alpha'], eps=optimizer_constructor.kwargs['eps'] )

	episodes_count = 1
	frames_per_episode = 1
	epsiodes_durations = []
	rewards_per_episode = 0
	rewards_duration = []
	loss_per_epoch = []
	wLoss_func = Weighted_Loss()

	
	current_state, _, _, _ = util.play_game(env, frames_per_state)
	print('Starting training...')

	for frames_count in range(1, max_frames):

		epsilon=exploreScheduler.anneal_linear(frames_count)
		beta = betaScheduler.anneal_linear(frames_count)
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

		td_error = torch.pow(torch.abs(td_error)+1e-8, prob_alpha)
		exp_replay.push(current_state, action, reward, curr_obs, td_error)
		current_state = curr_obs

		# compute y 
		if len(exp_replay) >= batch_size:
			# Get batch samples
			obs_samples, obs_ranks, obs_priorityVals = exp_replay.sample(batch_size)
			num_samples_per_batch = len(obs_samples)
			obs_priorityTensor = torch.from_numpy(np.array(obs_priorityVals))
			p_batch = 1/ obs_priorityTensor
			w_batch_raw = (1/len(exp_replay) * p_batch)**beta
			max_weight = exp_replay.get_max_weight(beta)
			w_batch = w_batch_raw/max_weight
			w_batch = w_batch.type(Tensor)
			
			batch = Experience(*zip(*obs_samples))

			loss, new_weights = ddqn_compute_y(batch, num_samples_per_batch, model, target, gamma, w_batch, wLoss_func)
			loss_abs = torch.abs(new_weights)
			exp_replay.update(obs_ranks, loss_abs)

			currentLOSS = loss.data.cpu().numpy()[0]

			# if np.isnan(currentLOSS) or np.isinf(currentLOSS):
			# 	print('NAN detected!!')
			# 	print('priority: ', obs_priorityVals)
			# 	print('max weight: ', max_weight)
			# 	print('weights beta: ', w_batch_raw)
			# 	print('norm weights: ', w_batch)

			optimizer.zero_grad()
			loss.backward()

			for param in model.parameters():
				param.grad.data.clamp_(-1,1)

			optimizer.step()
			loss_per_epoch.append(loss.data.cpu().numpy()[0])
		
		frames_per_episode+= frames_per_state

		if done:
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

		# sort memory replay every half of it's capacity iterations 
		if frames_count % int(rp_size/2) == 0:
			exp_replay.sort()


		#Save weights every 250k frames
		if frames_count % 250000 == 0:
			util.make_sure_path_exists(output_directory+'/')
			torch.save(model.state_dict(), output_directory+'/rank_weightedLoss_'+ str(frames_count)+'.pth')


		#Print frame count and sort experience replay for every 1000000 (one million) frames:
		if frames_count % 1000000 == 0:
			training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count, 'epsilon: ', epsilon
			print(training_update)
			logging.info(training_update)
			






