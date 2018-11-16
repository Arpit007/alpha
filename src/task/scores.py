import numpy as np
from src.task import correlation
from .pip import pip, mPip
from src.utils.misc import normalizeScore,normalizeScalarScore
import pandas as pd


def pearsonScore(user1, user2, userRatings, ratingTable, avgRating, gamma = 5):
	if user1 == user2:
		return 0.0
	
	# Get Correlated Item Ratings
	corr1 = []
	corr2 = []
	
	for key in userRatings.index:
		rating = ratingTable.loc[key, user2]
		if rating > 0:
			corr1.append(userRatings.loc[key])
			corr2.append(rating)
	
	if len(corr1) == 0:
		return 0.0
	
	# Calculate Score
	#user1Avg = np.average(corr1)
	#user2Avg = np.average(corr2)
	user1Avg = avgRating.loc[user1, 'avgRating']
	user2Avg = avgRating.loc[user2, 'avgRating']
	top = 0
	bottom1 = 0
	bottom2 = 0
	
	for r1, r2 in zip(corr1, corr2):
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
	penalty = min(len(corr1), gamma) / gamma
	
	# Penalise the Score
	penalisedScore = penalty * score
	
	return penalisedScore


def pearsonScores(userId, ratingTable, avgRating, gamma = 5):
	userRated = correlation.getUserRatedItems(ratingTable, userId)
	
	pScores = ratingTable.columns.map(lambda user: pearsonScore(userId, user, userRated, ratingTable, avgRating, gamma))
	
	return pScores


def calcAllPearson(ratingTable, avgRating):
	# Todo: Optimize
	pearsonScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pearsonScoresFrame[i] = pearsonScores(i, ratingTable, avgRating)
	
	return pearsonScoresFrame


def personalityScore(user1, user2, persScores):
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


def personalityScores(userId, ratingTable, persScores):
	pScores = ratingTable.columns.map(lambda user: personalityScore(userId, user, persScores))
	
	return pScores


def calcAllPersonalityScore(ratingTable, persScores):
	personalityScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		personalityScoresFrame[i] = personalityScores(i, ratingTable, persScores)
	
	return personalityScoresFrame


def pipScore(user1, user2, userRatings, ratingTable, itemsAvgRating):
	if user1 == user2:
		return 0
	
	score = 0
	
	for key in userRatings.index:
		r2 = ratingTable.loc[key, user2]
		if r2 > 0:
			r1=ratingTable.loc[key,user1]
			score+=pip(r1,r2,itemsAvgRating.loc[key,'avgRating'])
	
	return score

def pipScores(userId, ratingTable, itemsAvgRating):
	userRated = correlation.getUserRatedItems(ratingTable, userId)
	
	pScores = ratingTable.columns.map(lambda user: pipScore(userId, user, userRated, ratingTable, itemsAvgRating))
	
	return pScores

def calcAllPipScores(ratingTable, itemsAvgRating):
	pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pipScoresFrame[i] = pipScores(i, ratingTable, itemsAvgRating)
		#pipScoresFrame[i] = normalizeScore(pipScoresFrame[i])
	
	return pipScoresFrame
	
def calcAllNormalizedPipScores(ratingTable, itemsAvgRating):
	pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pipScoresFrame[i] = pipScores(i, ratingTable, itemsAvgRating)
		pipScoresFrame[i] = normalizeScore(pipScoresFrame[i])
	
	return pipScoresFrame
	
def calcAllNormalizedScalarPipScores(ratingTable, itemsAvgRating):
	pipScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pipScoresFrame[i] = pipScores(i, ratingTable, itemsAvgRating)
		pipScoresFrame[i] = normalizeScalarScore(pipScoresFrame[i])
	
	return pipScoresFrame

def mPipScore(r1, r2, rAvg, rMin = 1, rMax = 5):
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

def calcHybrid(pearsonScores, personalityScores, alpha = 0.4):
	return alpha * pearsonScores + (1 - alpha) * personalityScores
