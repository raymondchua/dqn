class Scheduler(object):
	def __init__(self, exploration_frame, initial_explore, final_explore):
		self.exploration_frame = exploration_frame
		self.initial_explore = initial_explore
		self.final_explore = final_explore

	def anneal_linear(self,t):
		if t <= self.exploration_frame:
			return self.initial_explore-((t/self.exploration_frame)*(self.initial_explore - self.final_explore))
		else:
			return self.final_explore