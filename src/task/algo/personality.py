import numpy as np
import pandas as pd


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


def personalityScore2Users(userId, ratingTable, persScores):
	pScores = ratingTable.columns.map(lambda user: personalityScore(userId, user, persScores))
	
	return pScores


def calculatePersonalityScores(ratingTable, persScores):
	personalityScoresFrame = pd.DataFrame(index = ratingTable.columns)
	for i in ratingTable.columns:
		personalityScoresFrame[i] = personalityScore2Users(i, ratingTable, persScores)
	
	return personalityScoresFrame