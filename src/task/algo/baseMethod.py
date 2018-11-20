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
		self.prediction = None
	
	@abstractmethod
	def calculate(self, ratingTable, avgRating, **params):
		pass
	
	def predict(self, ratingTable, avgRating, testRatingList, k):
		return predictTestRatings(testRatingList, ratingTable, self.score, avgRating, k)
	
	def predict_evaluate(self, ratingTable, avgRating, testRatingList, k, **params):
		pprint("Evaluating %s Method" % self.name)
		
		if self.score is None:
			self.calculate(ratingTable, avgRating, **params)
		
		self.prediction = self.predict(ratingTable, avgRating, testRatingList, k)
		
		testScores = metrics.specificity_precision_recall_accuracy(testRatingList['rating'], self.prediction)
		
		testScores.extend([metrics.mae(testRatingList['rating'], self.prediction),
		                   metrics.rmse(testRatingList['rating'], self.prediction)])
		
		self.metrics = testScores
