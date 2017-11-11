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
# import torchvision.transforms as T
# from PIL import Image

from replay_memory import ExpReplay, Experience
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

def dqn_compute_y(batch, batch_size, model, target, gamma):

	non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))) #to get a boolean value of 1 if not final 
	non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

	state_batch = Variable(torch.cat(batch.state)) #use cat to change data from tuple to tensor
	reward_batch = Variable(torch.cat(batch.reward)) 
	action_batch = Variable(torch.cat(batch.action))

	#compute Q(s,a) based on the action taken
	state_action_values = model(state_batch).gather(1,action_batch)

	next_state_action_values = Variable(torch.zeros(batch_size)).type(Tensor)
	next_state_action_values[non_final_mask] = target(non_final_next_states).max(1)[0]

	next_state_action_values.volatile = False

	y_output = reward_batch + (gamma * next_state_action_values)

	 # Compute Huber loss
	loss = F.smooth_l1_loss(state_action_values, y_output).clamp(-1,1)


	return loss



def dqn_train(env, scheduler, optimizer_constructor, model_type, batch_size, rp_start, rp_size, 
	exp_frame, exp_initial, exp_final, gamma, target_update_steps, frames_per_epoch, 
	frames_per_state, output_directory, last_checkpoint, envo):
	
	gym.undo_logger_setup()
	logging.basicConfig(filename=envo+'_'+model_type+'_training.log',level=logging.INFO)
	num_actions = env.action_space.n
	
	print('No. of actions: ', num_actions)
	print(env.unwrapped.get_action_meanings())

	# initialize action value and target network with the same weights
	model = DQN(num_actions, use_bn=False)
	target = DQN(num_actions, use_bn=False)

	if use_cuda:
		model.cuda()
		target.cuda()

	exp_replay = None
	episodes_count = 1


	if last_checkpoint != '':
		model.load_state_dict(torch.load(last_checkpoint))
		exp_replay = util.initialize_replay_resume(env, rp_start, rp_size, frames_per_state, model)
		episodes_count = get_index_from_checkpoint_path(last_checkpoint)

	else:
		exp_replay = util.initialize_replay(env, rp_start, rp_size, frames_per_state)

	target.load_state_dict(model.state_dict())
	print('weights loaded...')

	optimizer = optimizer_constructor.type(model.parameters(), lr=optimizer_constructor.kwargs['lr'],
		alpha=optimizer_constructor.kwargs['alpha'], eps=optimizer_constructor.kwargs['eps'] )
	
	frames_count = 1
	frames_per_episode = 1
	epsiodes_durations = []
	rewards_per_episode = 0
	rewards_duration = []
	loss_per_epoch = []

	env.reset()

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
		reward = Tensor([reward])

		exp_replay.push(current_state, action, reward, curr_obs)

		current_state = curr_obs

		#sample random mini-batch
		obs_sample = exp_replay.sample(batch_size)
		

		batch = Experience(*zip(*obs_sample)) #unpack the batch into states, actions, rewards and next_states

		#compute y 
		if len(exp_replay) >= batch_size:
			
			loss = dqn_compute_y(batch, batch_size, model, target, gamma)
			optimizer.zero_grad()
			loss.backward()

			for param in model.parameters():
				param.grad.data.clamp_(-1,1)

			optimizer.step()

			loss_per_epoch.append(loss.data.cpu().numpy()[0])
			

		frames_count+= 1
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
			# print('weights updated at frame no. ', frames_count)


		#Save weights every 250k frames
		if frames_count % 250000 == 0:
			util.make_sure_path_exists(output_directory+envo+'/'+model_type+'/')
			torch.save(model.state_dict(), output_directory+envo+'/'+model_type+'/weights_'+ str(frames_count)+'.pth')


		#Print frame count for every 1000000 (one million) frames:
		if frames_count % 1000000 == 0:
			training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count, 'epsilon: ', epsilon
			print(training_update)
			logging.info(training_update)






