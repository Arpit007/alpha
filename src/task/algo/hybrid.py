from src.task.algo.baseMethod import BaseMethod
from src.utils.printer import pprint


class Hybrid(BaseMethod):
	TASK = "hybrid"
	
	def __init__(self, ratingTable = None, avgRating = None, **params):
		super().__init__(Hybrid.TASK, "Hybrid")
		
		if ratingTable is not None:
			self.calculate(ratingTable, avgRating, **params)
	
	def calculate(self, ratingTable, avgRating, **params):
		self.name = "%s + %s" % (params["algo1"].name, params["algo2"].name)
		
		pprint('Calculating %s Scores' % self.name)
		
		self.score = params["alpha"] * params["algo1"].score + (1 - params["alpha"]) * params["algo2"].score
