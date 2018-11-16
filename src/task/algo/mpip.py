import pandas as pd

from src.task.algo.baseMethod import BaseMethod
from src.task.users import getUserRatedItems
from src.utils.misc import normalizeScore
from src.utils.misc import pprint


class MPip(BaseMethod):
	TASK = "mpip"
	
	def __init__(self, ratingTable = None, avgRating = None, **params):
		super().__init__(MPip.TASK, "M-PIP")
		
		if ratingTable is not None:
			self.calculate(ratingTable, avgRating, **params)
	
	def __mPip(self, r1, r2, rAvg, rMin = 1, rMax = 5):
		"""
			mPIP Function
			r1, r2: Any Two Ratings
			rAvg: Average Rating of Item by All Users
			rMin: Min Rating in the Rating Scale (like 1)
			rMax: Max Rating in the Rating Scale (like 5)
		"""
		rMed = 0.5
		r1 = 1 if r1 >= rMed else 0
		r2 = 1 if r2 >= rMed else 0
		rAvg = rAvg / rMax
		
		if ((r1 == 0 and r2 == 1) or (r1 == 1 and r2 == 0)):
			# False Agreement
			distance = 2
			impact = 1.0 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))  # Check
		else:
			# True Agreement
			distance = 0
			impact = (abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1)
		
		proximity = (3 - distance) ** 2
		if (r1 > rAvg and r2 > rAvg) or (r1 < rAvg and r2 < rAvg):
			popularity = 1 + (((r1 + r2) / 2) - rAvg) ** 2
		else:
			popularity = 1
		
		pipScore = proximity * impact * popularity
		return pipScore
	
	def __mPipScore(self, user1, user2, userRatings, ratingTable, itemsAvgRating):
		if user1 == user2:
			return 0
		
		score = 0
		
		for key in userRatings.index:
			r2 = ratingTable.loc[key, user2]
			if r2 > 0:
				r1 = ratingTable.loc[key, user1]
				score += self.__mPip(r1, r2, itemsAvgRating.loc[key, 'avgRating'])
		
		return score
	
	def __mPipScore2Users(self, userId, ratingTable, itemsAvgRating):
		userRated = getUserRatedItems(ratingTable, userId)
		
		pScores = ratingTable.columns.map(lambda user: self.__mPipScore(userId, user, userRated, ratingTable, itemsAvgRating))
		
		return pScores
	
	def calculate(self, ratingTable, avgRating, **params):
		pprint('Calculating %s Scores' % self.name)
		
		pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
		for i in ratingTable.columns:
			pipScoresFrame[i] = self.__mPipScore2Users(i, ratingTable, params["itemsAvgRating"])
			pipScoresFrame[i] = normalizeScore(pipScoresFrame[i])
		
		self.score = pipScoresFrame
