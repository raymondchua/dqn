import torch
import torch.nn as nn

class Weighted_Loss(nn.Module):
	def __init__(self):
		super(Weighted_Loss, self).__init__()

	def forward(self, current, target, weights):
		batch_loss = (torch.abs(current - target)<1).float()*(current - target)**2 +\
			(torch.abs(current - target)>=1).float()*(torch.abs(current - target) - 0.5)
		weighted_batch_loss = torch.dot(weights, batch_loss.squeeze())
		weighted_loss = weighted_batch_loss.sum()
		weighted_loss = torch.div(weighted_loss, current.size()[0])
		return weighted_loss, batch_loss