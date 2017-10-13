import numpy as np
from collections import namedtuple
import logging
import sys

import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ExpReplay
from dqn_model import DQN

from dqn_learn import Optimizer, dqn_inference
from util import Scheduler


BATCH_SIZE = 32
REPLAY_MEMO_SIZE = 100000
TARGET_UPDATE_FREQ = 10000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00025
INITIAL_EXPLORE = 1
FINAL_EXPLORE = 0.1
FRAMES_PER_EPOCH = 250000
EXPLORATION_FRAME = 1000000
REPLAY_START_SIZE = 50000

RMSPROP_ALPHA = 0.95
RMSPROP_EPS = 0.01

NUM_FRAMES_PER_STATE = 4

# REPLAY_START_SIZE = 500
# EXPLORATION_FRAME=10000
# FRAMES_PER_EPOCH = 10000




def main():

	env = gym.make('SpaceInvaders-v0').unwrapped
	scheduler = Scheduler(EXPLORATION_FRAME, INITIAL_EXPLORE, FINAL_EXPLORE)
	optimizer = Optimizer(type=optim.RMSprop, kwargs=dict(lr=LEARNING_RATE, alpha=RMSPROP_ALPHA, eps=RMSPROP_EPS))
	dqn_inference(env, scheduler, optimizer_constructor=optimizer, 
		batch_size = BATCH_SIZE, 
		rp_start=REPLAY_START_SIZE, 
		rp_size=REPLAY_MEMO_SIZE, 
		exp_frame=EXPLORATION_FRAME, 
		exp_initial=INITIAL_EXPLORE, 
		exp_final=FINAL_EXPLORE,
		gamma=DISCOUNT_FACTOR,
		target_update_steps=TARGET_UPDATE_FREQ,
		frames_per_epoch=FRAMES_PER_EPOCH,
		frames_per_state = NUM_FRAMES_PER_STATE)

	print("pass...")

if __name__ == '__main__':
	main()

