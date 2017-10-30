import os
import cv2
import gym
import random
import numpy as np
import errno

import torch
from torch.autograd import Variable

from replay_memory import ExpReplay
from rank_based_prioritized_replay import RankBasedPrioritizedReplay, Experience
from ddqn_rankPriority_learn import ddqn_compute_y, ddqn_compute_td_error

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def preprocessing_old(current_screen):

	current_screen_yuv = cv2.cvtColor(current_screen, cv2.COLOR_BGR2YUV)
	current_y, current_u, current_v = cv2.split(current_screen_yuv) #image size 210 x 160

	luminance = cv2.resize(current_y, (84,110)) #resize to 110 x 84
	luminance = luminance[21:-5,:] #remove the score

	return luminance

def get_screen_old(env):

	screen = env.render(mode='rgb_array')
	screen = preprocessing_old(screen)
	screen = np.expand_dims(screen, 0)
	
	return  torch.from_numpy(curr_state).unsqueeze(0).type(Tensor)

def play_game_old(env, num_frames, action=0, evaluate=False):

	state_reward = 0
	state_done = False
	state_obs = np.zeros((num_frames, 84, 84))

	for frame in range(num_frames):

		curr_obs, reward, done, _  = env.step(action)
		curr_obs_post = preprocessing_old(curr_obs)
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

	current_screen_yuv = cv2.cvtColor(current_screen, cv2.COLOR_BGR2YUV)
	current_y, current_u, current_v = cv2.split(current_screen_yuv) #image size 210 x 160

	current_y =  cv2.copyMakeBorder(current_y, 0,0, 25,25, cv2.BORDER_CONSTANT)

	luminance = cv2.resize(current_y, (84,84))

	return luminance

def get_screen(env):

	screen = env.render(mode='rgb_array')
	screen = preprocessing(screen)
	screen = np.expand_dims(screen, 0)
	
	return  torch.from_numpy(curr_state).unsqueeze(0).type(Tensor)

def play_game(env, num_frames, action=0, evaluate=False):

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

def get_Q_value(model, action, current_state):
	q_value = model(Variable(current_state, volatile=True).type(FloatTensor)).gather(1, action).data[0,0]
	return q_value

def get_greedy_action(model, current_state):
	output = model(Variable(current_state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1) #volatile = True means inference mode aka no learning
	return output

def initialize_replay(env, rp_start, rp_size, frames_per_state):
	exp_replay = ExpReplay(rp_size)
	episodes_count = 0
	env.reset()
	num_actions = env.action_space.n

	current_state, _, _, _ = play_game(env, frames_per_state)

	while episodes_count < rp_start:

		action = LongTensor([[random.randrange(num_actions)]])
		curr_obs, reward, done, _ = play_game(env, frames_per_state, action[0][0])
		reward = Tensor([reward])
		
		exp_replay.push(current_state, action, reward, curr_obs)

		current_state = curr_obs
		episodes_count+= 1

		if done:
			env.reset()
			current_state, _, _, _ = play_game(env, frames_per_state)
			

	print('Replay Memory initialized for training...')
	return exp_replay

def initialize_replay_resume(env, rp_start, rp_size, frames_per_state, model):
	exp_replay = ExpReplay(rp_size)
	episodes_count = 0
	env.reset()
	num_actions = env.action_space.n

	current_state, _, _, _ = play_game(env, frames_per_state)

	while episodes_count < rp_start:

		action = get_greedy_action(model, current_state)
		curr_obs, reward, done, _ = play_game(env, frames_per_state, action[0][0])
		reward = Tensor([reward])
		
		exp_replay.push(current_state, action, reward, curr_obs)

		current_state = curr_obs
		episodes_count+= 1

		if done:
			env.reset()
			current_state, _, _, _ = play_game(env, frames_per_state)
			

	print('Replay Memory re-initialized for training...')
	return exp_replay

def get_index_from_checkpoint_path(checkpoint):
	"""
	Get index from checkpoint filepath.
	Example of the filepath: /work/raymond/ddqn/saved_weights/ddqn_weights_9750000.pth
	"""
	key = checkpoint.split('/')
	chck_file = key[len(key)-1]
	chck_filename = chck_file.split('.')[0]
	chck_filename_key = chck_filename.split('_')
	chck_index = chck_filename_key[2]

	return int(chck_index)

def get_index_from_checkpoint_file(checkpoint):
	"""
	Get index from checkpoint file.
	Example of the filepath: ddqn_weights_9750000.pth
	"""
	chck_filename = checkpoint.split('.')[0]
	chck_filename_key = chck_filename.split('_')
	chck_index = chck_filename_key[2]

	return int(chck_index)

def initialize_rank_replay(env, rp_start, rp_size, frames_per_state, 
	model, target, gamma):

	exp_replay = RankBasedPrioritizedReplay(rp_size)
	episodes_count = 0
	env.reset()
	num_actions = env.action_space.n

	current_state, _, _, _ = play_game(env, frames_per_state)
	

	while episodes_count < rp_start:

		action = LongTensor([[random.randrange(num_actions)]])
		curr_obs, reward, done, _ = play_game(env, frames_per_state, action[0][0])
		reward = Tensor([[reward]])
		
		current_state_ex = Variable(current_state, volatile=True)
		curr_obs_ex = Variable(curr_obs, volatile=True)
		action_ex = Variable(action, volatile=True)
		reward_ex = Variable(reward, volatile=True)

		#compute td-error for one sample
		td_error = ddqn_compute_td_error(batch_size=1, state_batch=current_state_ex, reward_batch=reward_ex, action_batch=action_ex, 
			next_state_batch=curr_obs_ex, model=model, target=target, gamma=gamma)
		td_error = torch.abs(td_error)
		exp_replay.push(current_state, action, reward, curr_obs, td_error)

		current_state_ex = curr_obs_ex
		episodes_count+= 1

		if done:
			env.reset()
			current_state, _, _, _ = play_game(env, frames_per_state)


	exp_replay.sort()
	print('Rank Prioritized Replay initialized for training...')
	return exp_replay



# def initialize_rank_replay_resume(env, rp_start, rp_size, frames_per_state, 
# 	model, target, gamma, batch_size):
# 	exp_replay = RankBasedPrioritizedReplay(rp_size)
# 	episodes_count = 0
# 	env.reset()
# 	num_actions = env.action_space.n

# 	current_state, _, _, _ = play_game(env, frames_per_state)

# 	while episodes_count < rp_start:

# 		action = get_greedy_action(model, current_state)
# 		curr_obs, reward, done, _ = play_game(env, frames_per_state, action[0][0])
# 		reward = Tensor([[reward]])

# 		#compute td-error for one sample
# 		td_error = ddqn_compute_y(batch_size=1, batch=batch, model=model, target=target, gamma=gamma)
		
# 		exp_replay.push(current_state, action, reward, curr_obs, td_error)

# 		current_state = curr_obs
# 		episodes_count+= 1

# 		if done:
# 			env.reset()
# 			current_state, _, _, _ = play_game(env, frames_per_state)
			

# 	print('Rank Prioritized Replay re-initialized for training...')
# 	return exp_replay

def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
