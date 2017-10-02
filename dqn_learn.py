from __future__ import print_function, division

import cv2
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from replay_memory import ExpReplay, Experience
from dqn_model import DQN
from util import Scheduler

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
NO_OP_MAX = 8 #30 frames but since the frames are skipped randomly between 1 and 4
NO_OP_ACTION = 0

# MAX_FRAMES_PER_GAME = 50000

def get_screen(env):

	curr_state = np.zeros((4,84,84))

	for i in range(4):
		screen = env.render(mode='rgb_array')
		screen = preprocessing(screen)
		curr_state[i,:,:] = screen

	curr_state = curr_state / 255

	return torch.from_numpy(curr_state).unsqueeze(0)

def preprocessing(current_screen):

	current_screen_yuv = cv2.cvtColor(current_screen, cv2.COLOR_BGR2YUV)
	current_y, current_u, current_v = cv2.split(current_screen_yuv) #image size 210 x 160

	luminance = cv2.resize(current_y, (84,110)) #resize to 110 x 84
	luminance = luminance[26:,:] #remove the score

	return luminance

def initialize_replay(env, rp_start, rp_size):
	exp_replay = ExpReplay(rp_size)
	episodes_count = 0
	observation = env.reset()

	while episodes_count < rp_start:

		#create current state from observation
		current_state = get_screen(env)

		action = env.action_space.sample()
		_, reward, done, _ = env.step(action)

		if not done:
			next_state = get_screen(env)

		else:
			next_state = None
			observation = env.reset()

		#clip reward
		# reward = np.clip(reward, -1, 1)
		action = LongTensor([[action]])
		reward = Tensor([reward]).clamp(-1,1)
		# reward = Tensor([reward])

		exp_replay.push(current_state, action, reward, next_state)
		episodes_count+= 1
			

	print('Replay Memory Initialized.')
	return exp_replay

def compute_y(batch, batch_size, model, target, gamma):

	non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))) #to get a boolean value of 1 if not final 
	non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True).type(Tensor)

	state_batch = Variable(torch.cat(batch.state).type(Tensor)) #use cat to change data from tuple to tensor
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

def eval_model(env, model, epoch_count, eval_rand_init):
	eval_epsilon = 0.05
	num_actions = env.action_space.n
	env.reset()
	total_reward = []
	rewards_per_episode = 0

	action_value = torch.zeros(num_actions)

	for i in range(NUM_GAMES):

		for frame in range(MAX_FRAMES_PER_GAME):

			# different initial condition
			for no_op in range(eval_rand_init[i]):
				env.step(NO_OP_ACTION)

			eval_choice = random.uniform(0,1)
			curr_state = get_screen(env)

			# select a random action
			if eval_choice <= eval_epsilon:
				action = LongTensor([[random.randrange(num_actions)]])

			else:
				action = get_greedy_action(model, curr_state)

			_, reward, done, _ = env.step(action[0,0])

			# reward = np.clip(reward, -1, 1)

			action_value[action[0,0]] += get_Q_value(model, action.view(1,1), curr_state)

			rewards_per_episode += reward

			if done:
				# print('eval reward: ', rewards_per_episode)
				env.reset()
				total_reward.append(rewards_per_episode)
				rewards_per_episode = 0
				break

		

	average_reward = sum(total_reward)/float(len(total_reward))
	average_action_value = action_value/NUM_GAMES
	print('Model evaluated for epoch ', epoch_count)

	return average_reward, average_action_value

def get_Q_value(model, action, current_state):
	q_value = model(Variable(current_state, volatile=True).type(FloatTensor)).gather(1, action).data[0,0]
	return q_value

def get_greedy_action(model, current_state):

	output = model(Variable(current_state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1) #volatile = True means inference mode aka no learning
	return output

def dqn_inference(env, scheduler, optimizer_constructor=None, batch_size =16, rp_start=None, rp_size=None, 
	exp_frame=1000, exp_initial=1, exp_final=0.1, gamma=0.99, target_update_steps=1000, frames_per_epoch=10000):
	
	gym.undo_logger_setup()
	logging.basicConfig(filename='training.log',level=logging.INFO)

	observation = env.reset()
	exp_replay = initialize_replay(env, rp_start, rp_size)

	num_actions = env.action_space.n

	# initialize action value and target network with the same weights
	model = DQN(num_actions, use_bn=False)
	target = DQN(num_actions, use_bn=False)

	# model.load_state_dict(torch.load('./saved_weights/model_weights_10000.pth'))
	target.load_state_dict(model.state_dict())

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

	env.reset()
	curr_state = get_screen(env)

	eval_rand_init = np.random.randint(NO_OP_MAX, size=NUM_GAMES)


	print('Starting training...')

	while True:

		epsilon=scheduler.anneal_linear(frames_count)
		choice = random.uniform(0,1)

		# select a random action
		if choice <= epsilon:
			action = LongTensor([[random.randrange(num_actions)]])

		else:
			action = get_greedy_action(model, curr_state)

		_, reward, done, _ = env.step(action[0,0])

		if not done:
			next_state = get_screen(env)

		else:
			next_state = None

		reward = Tensor([reward]).clamp(-1,1)
		# reward = Tensor([reward])

		exp_replay.push(curr_state, action, reward, next_state)

		#sample random mini-batch
		obs_sample = exp_replay.sample(batch_size)
		batch = Experience(*zip(*obs_sample)) #unpack the batch into states, actions, rewards and next_states

		#compute y 
		loss = compute_y(batch, batch_size, model, target, gamma)
		optimizer.zero_grad()
		loss.backward()


		for param in model.parameters():
			if param.grad is not None:
				param.grad.data.clamp_(-1,1)

		optimizer.step()

		frames_count+= 1
		frames_per_episode+= 1

		if done:
			# epsiodes_durations.append(frames_per_episode)
			# if episodes_count % 10 == 0:
			# 	avg_episode_duration = sum(epsiodes_durations)/10.0
			# 	frames_per_episode_content = 'Episode from', episodes_count-10, ' to ', episodes_count, ' took ', avg_episode_duration, ' frames. '
			# 	print(frames_per_episode_content)
			# 	logging.info(frames_per_episode_content)
			# 	epsiodes_durations = []
			frames_per_episode=1
			episodes_count+=1
			env.reset()

		#update weights of target network for every TARGET_UPDATE_FREQ steps
		if frames_count % target_update_steps == 0:
			target.load_state_dict(model.state_dict())
			# print('weights updated at frame no. ', frames_count)

		#save weights of model for every 100000 frames_count:
		# if frames_count % 1000000 == 0:
		# 	torch.save(model.state_dict(), './saved_weights/model_weights_'+ str(frames_count)+'.pth')
		# 	print('weights saved at episode: ', episodes_count)


		#Run evaluation after each epoch and saved the weights
		if frames_count % frames_per_epoch == 0:
		# if episodes_count % 10 == 0:
			average_reward, average_action_value = eval_model(env, model, epoch_count, eval_rand_init)
			average_action_value = average_action_value.sum()/num_actions
			eval_content = 'Average Score for epoch ' + str(epoch_count) + ': ', average_reward
			average_action_value_content = 'Average Action Value for epoch ' + str(epoch_count) + ': ', average_action_value
			print(average_action_value_content)
			print(eval_content)
			logging.info(eval_content)
			logging.info(average_action_value_content)
			torch.save(model.state_dict(), './saved_weights/model_weights_'+ str(epoch_count)+'.pth')
			epoch_count += 1

		#Print frame count for every 1000000 (one million) frames:
		if frames_count % 1000000 == 0:
			training_update = 'frame count: ', frames_count, 'episode count: ', episodes_count
			print(training_update)
			logging.info(training_update)




