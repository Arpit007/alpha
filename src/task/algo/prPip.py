import numpy as np
import pandas as pd

from src.task.algo.baseMethod import BaseMethod
from src.utils.misc import normalizeScore, pprint


class PrPip(BaseMethod):
	TASK = "prPip"
	
	def __init__(self, ratingTable = None, avgRating = None, **params):
		super().__init__(PrPip.TASK, "PR-PIP")
		
		if ratingTable is not None:
			self.calculate(ratingTable, avgRating, **params)
	
	# Pip Score between two ratings of an Item
	def __pip(self, r1, r2, r1Avg, r2Avg, rMin = 1, rMax = 7):
		"""
		PIP Function
		r1, r2: Any Two Ratings
		rAvg: Average Rating of Item by All Users
		rMin: Min Rating in the Rating Scale (like 1)
		rMax: Max Rating in the Rating Scale (like 5)
		"""
		if r1 == 0 or r2 == 0:
			return 0
		
		rMed = 3.5
		
		if ((r1 > rMed and r2 < rMed) or (r1 < rMed and r2 > rMed)):
			# False Agreement
			distance = 2 * (r1 - r2)
			impact = 1.0 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
		else:
			# True Agreement
			distance = abs(r1 - r2)
			impact = ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
		
		proximity = (2 * (rMax - rMin) + 1 - distance) ** 2
		
		if (r1 > r1Avg and r2 > r2Avg) or (r1 < r1Avg or r2 < r2Avg):
			popularity = (1 + ((r1 + r2) / 2 - (r1Avg + r2Avg) / 2) ** 2)
		else:
			popularity = 1
		
		pipScore = proximity * impact * popularity
		return pipScore
	
	# Pip Score between 2 Users
	def __pipScore2Users(self, user1, user2, persScores):
		if user1 == user2:
			return 0
		
		score = 0
		
		u1Scores = persScores.loc[user1]
		u2Scores = persScores.loc[user2]
		
		user1Avg = np.average(u1Scores)
		user2Avg = np.average(u2Scores)
		
		for r1, r2 in zip(u1Scores, u2Scores):
			score += self.__pip(r1, r2, user1Avg, user2Avg)
		
		return score
	
	# Pip Score of a User with other Users
	def __pipScoreUsers(self, userId, ratingTable, persScores):
		
		pScores = ratingTable.columns.map(lambda user: self.__pipScore2Users(userId, user, persScores))
		
		return pScores
	
	def calculate(self, ratingTable, avgRating, **params):
		pprint('Calculating %s Scores' % self.name)
		
		pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
		for i in ratingTable.columns:
			pipScoresFrame[i] = self.__pipScoreUsers(i, ratingTable, params["persScores"])
			pipScoresFrame[i] = normalizeScore(pipScoresFrame[i])
		
		self.score = pipScoresFrame
