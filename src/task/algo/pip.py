from src.task import correlation
import pandas as pd
from src.utils.misc import normalizeScore


def pip(r1, r2, rAvg, rMin = 1, rMax = 5):
	"""
	PIP Function
	r1, r2: Any Two Ratings
	rAvg: Average Rating of Item by All Users
	rMin: Min Rating in the Rating Scale (like 1)
	rMax: Max Rating in the Rating Scale (like 5)
	"""
	if r1 == 0 or r2 == 0:
		return 0
	
	rMed = 2.5
	
	if ((r1 > rMed and r2 < rMed) or (r1 < rMed and r2 > rMed)):
		# False Agreement
		distance = 2 * (r1 - r2)
		impact = 1.0 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	else:
		# True Agreement
		distance = abs(r1 - r2)
		impact = ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	
	proximity = (2 * (rMax - rMin) + 1 - distance) ** 2
	
	if (r1 > rAvg and r2 > rAvg) or (r1 < rAvg or r2 < rAvg):
		popularity = (1 + ((r1 + r2) / 2 - rAvg) ** 2)
	else:
		popularity = 1
	
	pipScore = proximity * impact * popularity
	return pipScore


def pipScore(user1, user2, userRatings, ratingTable, itemsAvgRating):
	if user1 == user2:
		return 0
	
	score = 0
	
	for key in userRatings.index:
		r2 = ratingTable.loc[key, user2]
		if r2 > 0:
			r1 = ratingTable.loc[key, user1]
			score += pip(r1, r2, itemsAvgRating.loc[key, 'avgRating'])
	
	return score


def pipScore2Users(userId, ratingTable, itemsAvgRating):
	userRated = correlation.getUserRatedItems(ratingTable, userId)
	
	pScores = ratingTable.columns.map(lambda user: pipScore(userId, user, userRated, ratingTable, itemsAvgRating))
	
	return pScores


def calcPipScores(ratingTable, itemsAvgRating):
	pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pipScoresFrame[i] = pipScore2Users(i, ratingTable, itemsAvgRating)
		pipScoresFrame[i] = normalizeScore(pipScoresFrame[i])
	
	return pipScoresFrame
