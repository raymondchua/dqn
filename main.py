import numpy as np
from collections import namedtuple
import logging
import sys
import argparse
import os 


import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ExpReplay
from dqn_model import DQN

from dqn_learn import dqn_train
from dqn_eval import dqn_eval

from dqn_learn_old import dqn_train_old
from dqn_eval_old import dqn_eval_old

from ddqn_learn import ddqn_train
from ddqn_eval import ddqn_eval

from scheduler import Scheduler


# BATCH_SIZE = 32
# REPLAY_MEMO_SIZE = 100000
# TARGET_UPDATE_FREQ = 10000
# DISCOUNT_FACTOR = 0.99
# LEARNING_RATE = 0.00025
# INITIAL_EXPLORE = 1
# FINAL_EXPLORE = 0.1
# FRAMES_PER_EPOCH = 250000
# EXPLORATION_FRAME = 1000000
# REPLAY_START_SIZE = 50000

# RMSPROP_ALPHA = 0.95
# RMSPROP_EPS = 0.01

# NUM_FRAMES_PER_STATE = 4

# REPLAY_START_SIZE = 500
# EXPLORATION_FRAME=10000
# FRAMES_PER_EPOCH = 10000

parser = argparse = argparse.ArgumentParser(description='Deep Q Network Pytorch Implementation.')

parser.add_argument('--mode', 					type=str, 	help='train or eval', default='train')
parser.add_argument('--era', 					type=str, 	help='old or new', default='new')
parser.add_argument('--status', 				type=str,	help='Begin or resume. Begin meaning starts from zero.', default='begin')
parser.add_argument('--model_type', 			type=str,	help='Model architecture, eg. DQN', default='dqn', choices=['dqn', 'ddqn'])
parser.add_argument('--environment',			type=str,	help='Game environment, eg. SpaceInvaders-v0', default='SpaceInvaders-v0')
parser.add_argument('--input_size', 			type=int,	help='Input size for N x N. Resizing and/or padding is applied whenever necessary', default=84)
parser.add_argument('--batch_size',				type=int, 	help='Batch size', default=32)
parser.add_argument('--rp_capacity',			type=int, 	help='Replay Memory capacity', default=100000)
parser.add_argument('--rp_initial',				type=int, 	help='Initial size to populate Replay Memory', default=50000)
parser.add_argument('--target_update_steps',	type=int, 	help='The frequency with which the target network is updated', default=10000)
parser.add_argument('--frames_per_epoch',		type=int, 	help='Num frames per epoch. Useful as a counter for eval', default=250000)
parser.add_argument('--frames_per_state',		type=int, 	help='The number of most recent frames used as an input to the Q network. Actions are repeated over these frames.', default=4)
parser.add_argument('--discount_factor', 		type=float, help='Discount factor gamma used in the Q-learning udate', default=0.99)
parser.add_argument('--initial_explore', 		type=float,	help='Initial value of epsilon in epsilon-greedy exploration', default=1.0)
parser.add_argument('--final_explore', 			type=float,	help='Final value of epsilon in epsilon-greedy exploration', default=0.1)
parser.add_argument('--rmsprop_alpha',			type=float, help='Smoothing constant for RMSprop. See pytorch doc for more info.', default=0.95)
parser.add_argument('--rmsprop_eps',			type=float, help='Term added to the denominator to improve numerical stability for RMSprop.  See pytorch doc for more info.', default=0.01)
parser.add_argument('--explore_frame',			type=int, 	help='Num of frames over which the initial value of epsilon is linearly annealed to the final value', default=50000)
parser.add_argument('--learning_rate', 			type=float, help='Learning rate', default=0.00025)
parser.add_argument('--output_directory',		type=str,	help='Output directory to save weights, if empty, outputs to a local folder named \'saved_weights\'', default='./saved_weights/')
parser.add_argument('--last_checkpoint',		type=str,	help='Last saved weights that you wish to use to either resume training or for eval.', default='')

args = parser.parse_args()

Optimizer = namedtuple("Optimizer", ["type", "kwargs"])

def main():

	env = gym.make(args.environment).unwrapped
	scheduler = Scheduler(args.explore_frame, args.initial_explore, args.final_explore)
	optimizer = Optimizer(type=optim.RMSprop, kwargs=dict(lr=args.learning_rate, alpha=args.rmsprop_alpha, eps=args.rmsprop_eps))

	if args.model_type == 'dqn' and args.mode == 'train' and args.era == 'old':

		if not args.last_checkpoint:
			if not os.path.isfile(args.last_checkpoint):
				raise FileNotFoundError('Checkpoint file cannot be found!')

		dqn_train_old(env, scheduler, optimizer_constructor=optimizer, 
		model_type = args.model_type, 
		batch_size = args.batch_size, 
		rp_start = args.rp_initial, 
		rp_size = args.rp_capacity, 
		exp_frame = args.explore_frame, 
		exp_initial = args.initial_explore, 
		exp_final = args.final_explore,
		gamma = args.discount_factor,
		target_update_steps = args.target_update_steps,
		frames_per_epoch = args.frames_per_epoch,
		frames_per_state = args.frames_per_state,
		output_directory = args.output_directory,
		last_checkpoint = args.last_checkpoint)

	elif args.model_type == 'dqn' and args.mode == 'eval' and args.era == 'old':

		if not args.last_checkpoint:
			if not os.path.isfile(args.last_checkpoint):
				raise FileNotFoundError('Checkpoint file cannot be found!')

		dqn_eval_old(env, scheduler, optimizer_constructor=optimizer, 
		model_type = args.model_type, 
		batch_size = args.batch_size, 
		rp_start = args.rp_initial, 
		rp_size = args.rp_capacity, 
		exp_frame = args.explore_frame, 
		exp_initial = args.initial_explore, 
		exp_final = args.final_explore,
		gamma = args.discount_factor,
		target_update_steps = args.target_update_steps,
		frames_per_epoch = args.frames_per_epoch,
		frames_per_state = args.frames_per_state,
		output_directory = args.output_directory,
		last_checkpoint = args.last_checkpoint)

	elif args.model_type == 'dqn' and args.mode == 'train' and args.era == 'new':

		if not args.last_checkpoint:
			if not os.path.isfile(args.last_checkpoint):
				raise FileNotFoundError('Checkpoint file cannot be found!')

		dqn_train(env, scheduler, optimizer_constructor=optimizer, 
		model_type = args.model_type, 
		batch_size = args.batch_size, 
		rp_start = args.rp_initial, 
		rp_size = args.rp_capacity, 
		exp_frame = args.explore_frame, 
		exp_initial = args.initial_explore, 
		exp_final = args.final_explore,
		gamma = args.discount_factor,
		target_update_steps = args.target_update_steps,
		frames_per_epoch = args.frames_per_epoch,
		frames_per_state = args.frames_per_state,
		output_directory = args.output_directory,
		last_checkpoint = args.last_checkpoint)

	elif args.model_type == 'dqn' and args.mode == 'eval' and args.era == 'new':

		if not args.last_checkpoint:
			if not os.path.isfile(args.last_checkpoint):
				raise FileNotFoundError('Checkpoint file cannot be found!')


		dqn_eval(env, scheduler, optimizer_constructor=optimizer, 
		model_type = args.model_type, 
		batch_size = args.batch_size, 
		rp_start = args.rp_initial, 
		rp_size = args.rp_capacity, 
		exp_frame = args.explore_frame, 
		exp_initial = args.initial_explore, 
		exp_final = args.final_explore,
		gamma = args.discount_factor,
		target_update_steps = args.target_update_steps,
		frames_per_epoch = args.frames_per_epoch,
		frames_per_state = args.frames_per_state,
		output_directory = args.output_directory,
		last_checkpoint = args.last_checkpoint)

	elif args.model_type == 'ddqn' and args.mode == 'train' and args.era == 'new':

		if not args.last_checkpoint:
			if not os.path.isfile(args.last_checkpoint):
				raise FileNotFoundError('Checkpoint file cannot be found!')

		ddqn_train(env, scheduler, optimizer_constructor=optimizer, 
		model_type = args.model_type, 
		batch_size = args.batch_size, 
		rp_start = args.rp_initial, 
		rp_size = args.rp_capacity, 
		exp_frame = args.explore_frame, 
		exp_initial = args.initial_explore, 
		exp_final = args.final_explore,
		gamma = args.discount_factor,
		target_update_steps = args.target_update_steps,
		frames_per_epoch = args.frames_per_epoch,
		frames_per_state = args.frames_per_state,
		output_directory = args.output_directory,
		last_checkpoint = args.last_checkpoint)

	elif args.model_type == 'ddqn' and args.mode == 'eval' and args.era == 'new':

		if not args.last_checkpoint:
			if not os.path.isfile(args.last_checkpoint):
				raise FileNotFoundError('Checkpoint file cannot be found!')

		ddqn_eval(env, scheduler, optimizer_constructor=optimizer, 
		model_type = args.model_type, 
		batch_size = args.batch_size, 
		rp_start = args.rp_initial, 
		rp_size = args.rp_capacity, 
		exp_frame = args.explore_frame, 
		exp_initial = args.initial_explore, 
		exp_final = args.final_explore,
		gamma = args.discount_factor,
		target_update_steps = args.target_update_steps,
		frames_per_epoch = args.frames_per_epoch,
		frames_per_state = args.frames_per_state,
		output_directory = args.output_directory,
		last_checkpoint = args.last_checkpoint)




	# dqn_inference(env, scheduler, optimizer_constructor=optimizer, 
	# 	batch_size = BATCH_SIZE, 
	# 	rp_start=REPLAY_START_SIZE, 
	# 	rp_size=REPLAY_MEMO_SIZE, 
	# 	exp_frame=EXPLORATION_FRAME, 
	# 	exp_initial=INITIAL_EXPLORE, 
	# 	exp_final=FINAL_EXPLORE,
	# 	gamma=DISCOUNT_FACTOR,
	# 	target_update_steps=TARGET_UPDATE_FREQ,
	# 	frames_per_epoch=FRAMES_PER_EPOCH,
	# 	frames_per_state = NUM_FRAMES_PER_STATE)

	print("pass...")

if __name__ == '__main__':
	main()

