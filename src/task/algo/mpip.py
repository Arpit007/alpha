from src.task import correlation
import pandas as pd


def mPip(r1, r2, rAvg, rMin = 1, rMax = 5):
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


def mPipScore(user1, user2, userRatings, ratingTable, itemsAvgRating):
	if user1 == user2:
		return 0
	
	score = 0
	
	for key in userRatings.index:
		r2 = ratingTable.loc[key, user2]
		if r2 > 0:
			r1 = ratingTable.loc[key, user1]
			score += mPip(r1, r2, itemsAvgRating.loc[key, 'avgRating'])
	
	return score


def mPipScore2Users(userId, ratingTable, itemsAvgRating):
	userRated = correlation.getUserRatedItems(ratingTable, userId)
	
	pScores = ratingTable.columns.map(lambda user: mPipScore(userId, user, userRated, ratingTable, itemsAvgRating))
	
	return pScores


def calcMPipScores(ratingTable, itemsAvgRating):
	pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pipScoresFrame[i] = mPipScore2Users(i, ratingTable, itemsAvgRating)
	
	return pipScoresFrame
