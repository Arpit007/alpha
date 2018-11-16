from abc import ABC, abstractmethod

from src.task.rating import predictTestRatings
from src.utils import metrics
from src.utils.misc import pprint


class BaseMethod(ABC):
	def __init__(self, task, name):
		self.task = task
		self.name = name
		self.score = None
		self.metrics = None
	
	@abstractmethod
	def calculate(self, ratingTable, avgRating, **params):
		pass
	
	def predict_evaluate(self, ratingTable, avgRating, testRatingList, k, **params):
		pprint("Evaluating %s Method" % self.name)
		
		if self.score is None:
			self.calculate(ratingTable, avgRating, **params)
		
		# Todo: Why?
		testRatingList[self.task] = predictTestRatings(testRatingList, ratingTable, self.score, avgRating, k)
		
		testScores = metrics.specificity_precision_recall_accuracy(testRatingList['rating'], testRatingList[self.task])
		
		testScores.extend([metrics.mae(testRatingList['rating'], testRatingList[self.task]),
		                   metrics.rmse(testRatingList['rating'], testRatingList[self.task])])
		self.metrics = testScores
