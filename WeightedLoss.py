import torch
import torch.nn as nn

# class Weighted_Loss(nn.Module):
# 	def __init__(self):
# 		super(Weighted_Loss, self).__init__()

# 	def forward(self, current, target, weights):
# 		batch_loss = (torch.abs(current - target)<1).float()*(current - target)**2 +\
# 			(torch.abs(current - target)>=1).float()*(torch.abs(current - target) - 0.5)
# 		# weighted_batch_loss = torch.dot(weights, batch_loss.squeeze())
# 		weighted_loss = batch_loss.sum()
# 		weighted_loss = torch.div(weighted_loss, current.size()[0])
# 		return weighted_loss, batch_loss

class Weighted_Loss(nn.Module):
	"""
	Forward pass reference: https://goo.gl/rUVzj1
	"""


	def __init__(self):
		super(Weighted_Loss, self).__init__()


	def forward(self, input, target, size_average=True, reduce=True):
		abs_diff = (input - target).abs()
		ge_one_mask = (abs_diff >= 1).type_as(abs_diff)
		lt_one_mask = (abs_diff < 1).type_as(abs_diff)
		output = ge_one_mask * (abs_diff - 0.5) + lt_one_mask * 0.5 * (abs_diff ** 2)
		if reduce and size_average:
			return output.mean()
		elif reduce:
			return output.sum()
		return output