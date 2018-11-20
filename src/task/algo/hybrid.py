from src.task.algo.baseMethod import BaseMethod
from src.utils import metrics
from src.utils.misc import pprint


class Hybrid(BaseMethod):
	TASK = "hybrid"
	
	def __init__(self, ratingTable = None, avgRating = None, **params):
		super().__init__(Hybrid.TASK, "Hybrid")
		
		if ratingTable is not None:
			self.calculate(ratingTable, avgRating, **params)
	
	def calculate(self, ratingTable, avgRating, **params):
		self.name = "%s+%s" % (params["algo1"].name, params["algo2"].name)
		
		pprint('Calculating %s Scores' % self.name)
		
		self.alpha = params["alpha"]
		self.algo1 = params["algo1"]
		self.algo2 = params["algo2"]
	
	def predict_evaluate(self, ratingTable, avgRating, testRatingList, k, **params):
		self.prediction = self.algo1.prediction * self.alpha + self.algo2.prediction * (1 - self.alpha)
		
		testScores = metrics.specificity_precision_recall_accuracy(testRatingList['rating'], self.prediction)
		
		testScores.extend([metrics.mae(testRatingList['rating'], self.prediction),
		                   metrics.rmse(testRatingList['rating'], self.prediction)])
		
		self.metrics = testScores
