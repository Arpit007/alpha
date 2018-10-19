import numpy as np
from src.task import correlation
import pandas as pd


def pearsonScore(user1, user2, userRatings, ratingTable, gamma = 5):
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
	user1Avg = np.average(corr1)
	user2Avg = np.average(corr2)
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


def pearsonScores(userId, ratingTable, gamma = 5):
	userRated = correlation.getUserRatedItems(ratingTable, userId)
	
	pScores = ratingTable.columns.map(lambda user: pearsonScore(userId, user, userRated, ratingTable, gamma))
	
	return pScores


def calcAllPearson(ratingTable):
	# Todo: Optimize
	pearsonScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		pearsonScoresFrame[i] = pearsonScores(i, ratingTable)
	
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


def calcPearsonPersonality(pearsonScores, personalityScores, alpha = 0.4):
	return alpha * pearsonScores + (1 - alpha) * personalityScores
