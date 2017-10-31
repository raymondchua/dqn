import torch
import torch.nn as nn
import torch.nn.functional as F


class DUEL(nn.Module):
	def __init__(self, num_actions, use_bn=False):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)	
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
		
		self.value1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
		self.value2 = nn.Linear(in_features=512, out_features=1)
		
		self.advantage1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
		self.advantage2 = nn.Linear(in_features=512, out_features=num_actions)
	

	def forward(self, x):

		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		
		x_value = F.relu(self.value1(x.view(x.size(0), -1)))
		x_value = self.value2(x_value.view(x_value.size(0), -1))

		x_advantage = F.relu(self.advantage1(x.view(x.size(0), -1)))
		x_advantage = self.advantage2(x_advantage.view(x_advantage.size(0), -1))
		
		return x_value.add(x_advantage - x_advantage.sum()/num_actions)