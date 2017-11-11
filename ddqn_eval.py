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
import os


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
from util import *


import time

#(BCHW)

# Optimizer = namedtuple("Optimizer", ["type", "kwargs"])

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

def ddqn_compute_y(batch, batch_size, model, target, gamma):

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
	loss = F.smooth_l1_loss(state_action_values, y_output)

	return loss

def ddqn_eval(env, scheduler, optimizer_constructor, model_type, batch_size, rp_start, rp_size, 
	exp_frame, exp_initial, exp_final, gamma, target_update_steps, frames_per_epoch, 
	frames_per_state, output_directory, last_checkpoint):
	
	env.seed(7)
	random.seed(7)
	gym.undo_logger_setup()
	logging.basicConfig(filename=envo+'_'+'ddqn_eval.log',level=logging.INFO)
	num_actions = env.action_space.n
	
	print('No. of actions: ', num_actions)
	print(env.unwrapped.get_action_meanings())

	# initialize action value and target network with the same weights
	model = DQN(num_actions, use_bn=False)

	if use_cuda:
		model.cuda()


	saved_params = None
	directory = None

	index = []

	for (dirpath, dirnames, filenames) in os.walk(output_directory):
		directory = dirpath
		saved_params = filenames

	count = 0 
	counter = 0

	chckpoint_index = get_index_from_checkpoint_path(last_checkpoint)

	for x in saved_params:
		temp = get_index_from_checkpoint_file(x)
		if temp > chckpoint_index:
			index.append(temp)

	index = sorted(index, key=int)

	for w in index:
		path = directory  + model_type + '_weights_' + str(w) + '.pth'
		model.load_state_dict(torch.load(path))
		print(path)
		print('saved weights loaded...')

		eval_epsilon = 0.05
		env.reset()
		total_reward = []
		rewards_per_episode = 0

		action_value = torch.zeros(num_actions)

		current_state, _, _, _ = play_game(env, frames_per_state, action=0, evaluate=True)

		average_action = {k: [] for k in range(num_actions)}

		for i in range(NUM_GAMES):
			for frame in range(int(MAX_FRAMES_PER_GAME/frames_per_state)):

				eval_choice = random.uniform(0,1)
	
				# select a random action
				if eval_choice <= eval_epsilon:
					action = LongTensor([[random.randrange(num_actions)]])

				else:
					action = get_greedy_action(model, current_state)

				# _, reward, done, _ = env.step(action[0,0])
				curr_obs, reward, done, _ = play_game(env, frames_per_state, action[0][0], evaluate=True)

				average_action[action[0,0]].append(get_Q_value(model, action.view(1,1), curr_obs))

				current_state = curr_obs
				rewards_per_episode += reward

				if done:
					env.reset()
					total_reward.append(rewards_per_episode)
					rewards_per_episode = 0
					current_state, _, _, _ = play_game(env, frames_per_state, action=0, evaluate=True)
					break

		average_reward = sum(total_reward)/float(len(total_reward))

		total_action = 0
		for i in range(num_actions):
			total_action += sum(average_action[i])/len(average_action[i])

		average_action_value = total_action/num_actions

		#Compute Standard Deviation
		diff = 0
		for x in total_reward:
			diff += (x - average_reward)*(x - average_reward)
		var = diff/len(total_reward)
		std_dev = math.sqrt(var)


		eval_content = 'Average Score: ', average_reward
		eval_std_dev = 'Standard Deviation: ', std_dev
		average_action_value_content = 'Average Action Value: ', average_action_value
		print(average_action_value_content)
		print(eval_content)
		print(eval_std_dev)
		log_content = path + ' ' + str(average_reward) + ' ' + str(average_action_value) + ' ' + str(std_dev)
		logging.info(log_content)

		count += 1

	print(count)
