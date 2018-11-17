import numpy as np
import pandas as pd

from src.task.algo.baseMethod import BaseMethod
from src.utils.misc import pprint


class Personality(BaseMethod):
	TASK = "personality"
	
	def __init__(self, ratingTable = None, avgRating = None, **params):
		super().__init__(Personality.TASK, "Prsnlty")
		
		if ratingTable is not None:
			self.calculate(ratingTable, avgRating, **params)
	
	# Personality Score between 2 Users
	def __personalityScore(self, user1, user2, persScores):
		if user1 == user2:
			return 0.0
		
		u1Scores = persScores.loc[user1]
		u2Scores = persScores.loc[user2]
		
		user1Avg = np.average(u1Scores)
		user2Avg = np.average(u2Scores)
		
		top = 0
		bottom1 = 0
		bottom2 = 0
		
		for r1, r2 in zip(u1Scores, u2Scores):
			tr1 = r1 - user1Avg
			tr2 = r2 - user2Avg
			top += tr1 * tr2
			bottom1 += tr1 ** 2
			bottom2 += tr2 ** 2
		
		bottom = bottom1 * bottom2
		bottom = np.sqrt(bottom)
		
		if bottom == 0:
			return 0.0
		
		score = top / bottom
		
		return score
	
	# Personality Score of a User with other Users
	def __personalityScoreUsers(self, userId, ratingTable, persScores):
		pScores = ratingTable.columns.map(lambda user: self.__personalityScore(userId, user, persScores))
		
		return pScores
	
	def calculate(self, ratingTable, avgRating, **params):
		pprint('Calculating %s Scores' % self.name)
		personalityScoresFrame = pd.DataFrame(index = ratingTable.columns)
		for i in ratingTable.columns:
			personalityScoresFrame[i] = self.__personalityScoreUsers(i, ratingTable, params["persScores"])
		
		self.score = personalityScoresFrame
