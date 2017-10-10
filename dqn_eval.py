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

from replay_memory import ExpReplay, Experience
from dqn_model import DQN
from util import Scheduler


import time

#(BCHW)

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
NO_OP_MAX = 30
NO_OP_ACTION = 0

# MAX_FRAMES_PER_GAME = 50000

resize = T.Compose([T.ToPILImage(),
					T.Scale((84,110), interpolation=Image.BILINEAR),
					T.ToTensor()])

# count = 0

def get_screen(env):

	# curr_state = np.zeros((4,84,84))

	# screen = env.render(mode='rgb_array').transpose((2, 0, 1))
	
	# screen = screen[:,26:,:]
	# screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	# screen = torch.from_numpy(screen)

	screen = env.render(mode='rgb_array')
	screen = preprocessing(screen)
	screen = np.expand_dims(screen, 0)
	# curr_state[i,:,:] = screen

	# for i in range(4):
	# 	screen = env.render(mode='rgb_array')
	# 	screen = preprocessing(screen)
	# 	curr_state[i,:,:] = screen

	# curr_state = curr_state / 255

	# return resize(screen).unsqueeze(0).type(Tensor)
	return  torch.from_numpy(curr_state).unsqueeze(0).type(Tensor)

def play_game(env, num_frames, model, num_actions, action=0, evaluate=False):

	state_reward = 0
	state_done = False
	state_obs = np.zeros((num_frames, 84, 84))

	for frame in range(num_frames):

		curr_obs, reward, done, _  = env.step(action)
		curr_obs_post = preprocessing(curr_obs)
		state_obs[frame,:,:] = curr_obs_post
		state_done = state_done | done
		state_reward += reward

		
	if state_done:
		state_reward += -1 

	if state_reward < -1 and not evaluate:
		state_reward = -1

	elif state_reward > 1 and not evaluate:
		state_reward = 1

	state_obs = state_obs / 255
	state_obs = torch.from_numpy(state_obs).unsqueeze(0).type(Tensor)



	return state_obs, state_reward, state_done, _


def preprocessing(current_screen):

	# global count

	current_screen_yuv = cv2.cvtColor(current_screen, cv2.COLOR_BGR2YUV)
	current_y, current_u, current_v = cv2.split(current_screen_yuv) #image size 210 x 160

	luminance = cv2.resize(current_y, (84,110)) #resize to 110 x 84
	luminance = luminance[21:-5,:] #remove the score

	# cv2.imwrite('./images/image_'+str(count)+'.png',luminance)
	# count+= 1

	return luminance

def initialize_replay(env, exp_initial, rp_start, rp_size, num_actions, frames_per_state, model):
	exp_replay = ExpReplay(rp_size)
	episodes_count = 0
	env.reset()

	current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions)


	while episodes_count < rp_start:

		action = LongTensor([[random.randrange(num_actions)]])
		curr_obs, reward, done, _ = play_game(env, frames_per_state, model, num_actions, action[0][0])
		reward = Tensor([reward])
		
		exp_replay.push(current_state, action, reward, curr_obs)

		current_state = curr_obs
		episodes_count+= 1

		if done:
			env.reset()
			current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions)
			

	print('Replay Memory Initialized.')
	return exp_replay

def compute_y(batch, batch_size, model, target, gamma):

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

def eval_model(env, model, epoch_count, eval_rand_init, frames_per_state):
	eval_epsilon = 0.05
	num_actions = env.action_space.n
	env.reset()
	total_reward = []
	rewards_per_episode = 0

	action_value = torch.zeros(num_actions)

	current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions, action=0, evaluate=True)

	for i in range(NUM_GAMES):


		for frame in range(int(MAX_FRAMES_PER_GAME/frames_per_state)):

			# different initial condition
			# for no_op in range(eval_rand_init[i]):
			# 	env.step(NO_OP_ACTION)

			eval_choice = random.uniform(0,1)

			# select a random action
			if eval_choice <= eval_epsilon:
				action = LongTensor([[random.randrange(num_actions)]])

			else:
				action = get_greedy_action(model, current_state)

			# _, reward, done, _ = env.step(action[0,0])
			curr_obs, reward, done, _ = play_game(env, frames_per_state, model, num_actions, action[0][0], evaluate=True)

			action_value[action[0,0]] += get_Q_value(model, action.view(1,1), curr_obs)

			current_state = curr_obs

			rewards_per_episode += reward

			if done:
				# print('eval reward: ', rewards_per_episode)
				env.reset()
				total_reward.append(rewards_per_episode)
				rewards_per_episode = 0
				current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions, action=0, evaluate=True)
				break

	average_reward = sum(total_reward)/float(len(total_reward))
	average_action_value = action_value.numpy()/NUM_GAMES
	print('Model evaluated for epoch ', epoch_count)

	return average_reward, average_action_value

def get_Q_value(model, action, current_state):
	q_value = model(Variable(current_state, volatile=True).type(FloatTensor)).gather(1, action).data[0,0]
	return q_value

def get_greedy_action(model, current_state):

	output = model(Variable(current_state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1) #volatile = True means inference mode aka no learning
	return output

def dqn_inference(env, scheduler, optimizer_constructor=None, batch_size =16, rp_start=None, rp_size=None, 
	exp_frame=1000, exp_initial=1, exp_final=0.1, gamma=0.99, target_update_steps=1000, frames_per_epoch=10000, 
	frames_per_state=4):
	
	gym.undo_logger_setup()
	logging.basicConfig(filename='dqn_training.log',level=logging.INFO)
	num_actions = env.action_space.n
	
	print('No. of actions: ', num_actions)
	print(env.unwrapped.get_action_meanings())

	# initialize action value and target network with the same weights
	model = DQN(num_actions, use_bn=False)
	target = DQN(num_actions, use_bn=False)

	# model.load_state_dict(torch.load('./saved_weights/neg_model_weights_6000.pth'))
	target.load_state_dict(model.state_dict())

	exp_replay = initialize_replay(env, exp_initial, rp_start, rp_size, num_actions, frames_per_state, model)

	print('weights loaded...')

	if use_cuda:
		model.cuda()
		target.cuda()

	scheduler = Scheduler(exp_frame, exp_initial, exp_final)
	optimizer = optimizer_constructor.type(model.parameters(), lr=optimizer_constructor.kwargs['lr'],
		alpha=optimizer_constructor.kwargs['alpha'], eps=optimizer_constructor.kwargs['eps'] )


	episodes_count = 1
	frames_count = 1
	epoch_count = 1
	frames_per_episode = 1
	epsiodes_durations = []
	rewards_per_episode = 0
	rewards_duration = []
	loss_per_epoch = []

	env.reset()
	current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions)

	eval_rand_init = np.random.randint(NO_OP_MAX, size=NUM_GAMES)
	print(eval_rand_init)

	print('Starting training...')

	count = 0

	while True:

		# curr_state = get_screen(env)
		epsilon=scheduler.anneal_linear(frames_count)
		choice = random.uniform(0,1)

		# select a random action
		if choice <= epsilon:
			action = LongTensor([[random.randrange(num_actions)]])

		else:
			action = get_greedy_action(model, current_state)

		
		curr_obs, reward, done, _ = play_game(env, frames_per_state, model, num_actions, action[0][0])

		rewards_per_episode += reward
		reward = Tensor([reward])

		exp_replay.push(current_state, action, reward, curr_obs)

		current_state = curr_obs

		#sample random mini-batch
		obs_sample = exp_replay.sample(batch_size)

		batch = Experience(*zip(*obs_sample)) #unpack the batch into states, actions, rewards and next_states

		#compute y 
		if len(exp_replay) >= batch_size:
			
			loss = compute_y(batch, batch_size, model, target, gamma)
			optimizer.zero_grad()
			loss.backward()

			for param in model.parameters():
				param.grad.data.clamp_(-1,1)

			optimizer.step()
			loss_per_epoch.append(loss.data.cpu().numpy())

		frames_count+= 1
		frames_per_episode+= frames_per_state

		if done:
			# epsiodes_durations.append(frames_per_episode)
			rewards_duration.append(rewards_per_episode)
			# rewards_per_episode_content = 'Episode: ', episodes_count, 'Reward: ', rewards_per_episode
			# logging.info(rewards_per_episode_content)
			rewards_per_episode = 0
			frames_per_episode=1
			episodes_count+=1
			env.reset()
			current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions)

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


		#Run evaluation after each epoch and saved the weights
		# if frames_count % frames_per_epoch == 0:
		if frames_count % 250000 == 0:
			average_reward, average_action_value = eval_model(env, model, epoch_count, eval_rand_init, frames_per_state)
			# average_action_value = average_action_value.sum()/num_actions
			eval_content = 'Average Score for epoch ' + str(epoch_count) + ': ', average_reward
			average_action_value_content = 'Average Action Value for epoch ' + str(epoch_count) + ': ', average_action_value
			print(average_action_value_content)
			print(eval_content)
			logging.info(eval_content)
			logging.info(average_action_value_content)
			torch.save(model.state_dict(), './saved_weights/dqn_weights_'+ str(frames_count)+'.pth')
			epoch_count += 1

		#Print frame count for every 1000000 (one million) frames:
		if frames_count % 1000000 == 0:
			training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count, 'epsilon: ', epsilon
			print(training_update)
			logging.info(training_update)


def dqn_eval(env, scheduler, optimizer_constructor=None, batch_size =16, rp_start=None, rp_size=None, 
	exp_frame=1000, exp_initial=1, exp_final=0.1, gamma=0.99, target_update_steps=1000, frames_per_epoch=10000, 
	frames_per_state=4):
	
	gym.undo_logger_setup()
	logging.basicConfig(filename='dqn_eval.log',level=logging.INFO)
	num_actions = env.action_space.n
	
	print('No. of actions: ', num_actions)
	print(env.unwrapped.get_action_meanings())

	# initialize action value and target network with the same weights
	model = DQN(num_actions, use_bn=False)


	model.load_state_dict(torch.load('./saved_weights/dqn_weights_5000000.pth'))
	print('saved weights loaded...')

	if use_cuda:
		model.cuda()

	eval_epsilon = 0.05
	env.reset()
	total_reward = []
	rewards_per_episode = 0

	# eval_rand_init = np.random.randint(NO_OP_MAX, size=NUM_GAMES)
	# print(eval_rand_init)

	action_value = torch.zeros(num_actions)

	current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions, action=0, evaluate=True)

	average_action = {k: [] for k in range(num_actions)}

	for i in range(NUM_GAMES):
		for frame in range(int(MAX_FRAMES_PER_GAME/frames_per_state)):

			# different initial condition
			# for no_op in range(eval_rand_init[i]):
			# 	env.step(NO_OP_ACTION)

			eval_choice = random.uniform(0,1)

			# select a random action
			if eval_choice <= eval_epsilon:
				action = LongTensor([[random.randrange(num_actions)]])

			else:
				action = get_greedy_action(model, current_state)

			# _, reward, done, _ = env.step(action[0,0])
			curr_obs, reward, done, _ = play_game(env, frames_per_state, model, num_actions, action[0][0], evaluate=True)

			# action_value[action[0,0]] += get_Q_value(model, action.view(1,1), curr_obs)
			average_action[action[0,0]].append(get_Q_value(model, action.view(1,1), curr_obs))

			current_state = curr_obs

			rewards_per_episode += reward

			if done:
				env.reset()
				print(rewards_per_episode)
				total_reward.append(rewards_per_episode)
				rewards_per_episode = 0
				current_state, _, _, _ = play_game(env, frames_per_state, model, num_actions, action=0, evaluate=True)
				break

	average_reward = sum(total_reward)/float(len(total_reward))

	total_action = 0
	for i in range(num_actions):
		total_action += sum(average_action[i])/len(average_action[i])

	average_action_value = total_action/num_actions


	eval_content = 'Average Score: ', average_reward
	average_action_value_content = 'Average Action Value: ', average_action_value
	print(average_action_value_content)
	print(eval_content)
	logging.info(eval_content)
	logging.info(average_action_value_content)

