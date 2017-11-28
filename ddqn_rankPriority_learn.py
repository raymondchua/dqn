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
	
def ddqn_compute_y(batch, batch_size, model, target, gamma, weights, lossFunc):
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

	#compute Q(s,a) based on the action taken
	state_action_values = model(state_batch).gather(1,action_batch)

	model_actions = model(non_final_next_states).data.max(1)[1].view(batch_size,1)
	model_action_batch = Variable(torch.cat([model_actions]), volatile=True)

	next_state_action_values = Variable(torch.zeros(batch_size)).type(Tensor)
	next_state_action_values[non_final_mask] = target(non_final_next_states).gather(1, model_action_batch)
	next_state_action_values.volatile = False

	y_output =  (next_state_action_values*gamma) + reward_batch.squeeze()
	y_output = y_output.view(batch_size,1)
	
	# Compute Huber loss
	# loss = F.smooth_l1_loss(state_action_values, y_output)

	weights_var = Variable(weights)
	# loss, td_error = lossFunc(state_action_values, y_output, weights_var)
	# loss = F.smooth_l1_loss(state_action_values, y_output, size_average=False)
	# loss = lossFunc(state_action_values, y_output, weights_var, reduce=False).squeeze()

	loss = torch.squeeze(y_output - state_action_values)
	loss = torch.clamp(loss, -1, 1)

	wloss = torch.dot(loss, weights_var)
	avgLoss = wloss.mean()

	# loss.data.clamp_(-1,1)
	td_error = torch.abs(loss.data)
	td_error = td_error + 1e-8
	
	return avgLoss, td_error

	

def ddqn_rank_train(env, exploreScheduler, betaScheduler, optimizer_constructor, model_type, batch_size, rp_start, rp_size, 
	exp_frame, exp_initial, exp_final, prob_alpha, gamma, target_update_steps, frames_per_epoch, 
	frames_per_state, output_directory, last_checkpoint, max_frames, envo):

	"""
	Implementation of the training algorithm for DDQN using Rank-based prioritization.
	Information with regards to the algorithm can be found in the paper, 
	"Prioritized Experience Replay" by Tom Schaul, John Quan, Ioannis Antonoglou and
	David Silver. Refer to section 3.3 in the paper for more info. 
	"""
	
	gym.undo_logger_setup()
	logging.basicConfig(filename=envo+'_'+'ddqn_rank_weighted_training.log',level=logging.INFO)
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

		#TODO: Implementation of resume
		# exp_replay = util.initialize_rank_replay_resume(env, rp_start, rp_size, frames_per_state, 
		# 	model, target, gamma, batch_size)
		# frames_count = get_index_from_checkpoint_path(last_checkpoint)

	else:
		exp_replay = util.initialize_rank_replay(env, rp_start, rp_size, frames_per_state, 
			model, target, gamma, prob_alpha)

	target.load_state_dict(model.state_dict())

	optimizer = optimizer_constructor.type(model.parameters(), lr=optimizer_constructor.kwargs['lr'],
		alpha=optimizer_constructor.kwargs['alpha'], eps=optimizer_constructor.kwargs['eps'] )

	episodes_count = 1
	epsiodes_durations = []
	rewards_per_episode = 0
	rewards_duration = []
	loss_per_epoch = []
	current_state, _, _, _ = util.play_game(env, frames_per_state)
	wLoss_func = Weighted_Loss()

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
		td_error = 1

		temp_exp = Experience(current_state, action, reward, curr_obs, td_error)
		current_state = curr_obs

		# compute y 
		if len(exp_replay) >= batch_size:
			# Get batch samples

			# start = time.time()

			if frames_count%rp_size==0:
				obs_samples, obs_ranks, obs_priorityVals = exp_replay.sample(batch_size-1, prob_alpha ,sort=True)
			else:
				obs_samples, obs_ranks, obs_priorityVals = exp_replay.sample(batch_size-1, prob_alpha, sort=False)

			obs_samples.append(temp_exp)
			obs_priorityVals.append(td_error)

			obs_pVals_tensor = torch.from_numpy(np.array(obs_priorityVals))
			print("P(i): ", obs_pVals_tensor)
			IS_weights = torch.pow((obs_pVals_tensor * rp_size), -beta)
			max_weight = torch.max(IS_weights)

			print("W(i): ", IS_weights)
			IS_weights_norm = torch.div(IS_weights, max_weight).type(Tensor)
			IS_weights_norm[-1] = torch.max(IS_weights_norm)

			batch = Experience(*zip(*obs_samples))
			loss, new_weights = ddqn_compute_y(batch, batch_size, model, target, gamma, IS_weights_norm, wLoss_func)
			new_weights = torch.pow(new_weights, prob_alpha)
			new_exp = Experience(temp_exp.state, temp_exp.action, temp_exp.reward, temp_exp.next_state, new_weights[batch_size-1])
			exp_replay.update(obs_ranks, new_weights, new_exp)
			optimizer.zero_grad()
			loss.backward()

			optimizer.step()
			loss_per_epoch.append(loss.data.cpu().numpy()[0])

		else:
			exp_replay.push(new_exp.state, new_exp.action, new_exp.reward, new_exp.next_state, td_error)



		# end = time.time()

		# duration = end-start

		# print('duration : ', duration)

		if done:
			# print('Game: ', rewards_per_episode)
			rewards_duration.append(rewards_per_episode)
			rewards_per_episode = 0
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


		#Save weights every 250k frames
		if frames_count % 250000 == 0:
			util.make_sure_path_exists(output_directory+'/'+envo+'/')
			torch.save(model.state_dict(), output_directory+'/'+envo+'/rank_uniform'+ str(frames_count)+'.pth')


		#Print frame count and sort experience replay for every 1000000 (one million) frames:
		if frames_count % 1000000 == 0:
			training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count, 'epsilon: ', epsilon
			print(training_update)
			logging.info(training_update)
			






