import numpy as np
import pandas as pd


def getKappaFactor(scores):
	summation = np.sum(np.abs(scores))
	if summation == 0:
		return 0
	kappa = 1 / summation
	return kappa


def getUsersAverageRating(ratingTable):
	avgRatings = pd.DataFrame(ratingTable.apply(lambda x: np.average(x[x != 0])), columns = ['avgRating'])
	avgRatings.insert(0, 'userId', avgRatings.index)
	return avgRatings


def predictRating(userId, itemId, ratingTable, scoreTable, avgRating, k = 5):
	tRatingTable = ratingTable.T
	usersItemRating = tRatingTable[tRatingTable[itemId] > 0][[itemId]]
	usersItemRating = usersItemRating.rename(columns = { itemId: 'rating' })
	usersItemRating = usersItemRating.reset_index()
	
	scores = scoreTable[[userId]]
	scores = scores.rename(columns = { userId: 'score' })
	scores = scores.reset_index()
	
	itemRatingList = pd.merge(usersItemRating, scores, on = 'userId', how = 'left')
	itemRatingList = pd.merge(itemRatingList, avgRating, on = 'userId', how = 'left')
	itemRatingList = itemRatingList.sort_values('score', ascending = False)[:k]
	
	kappa = getKappaFactor(itemRatingList['score'])
	currUserAvg = avgRating.loc[userId, 'avgRating']
	tempVal = itemRatingList['score'] * (itemRatingList['rating'] - itemRatingList['avgRating'])
	rating = currUserAvg + kappa * np.sum(tempVal)
	
	return rating


def getOrCalculateUserItemRating(userId, userElements, ratingTable, scoreTable, avgRatings, k = 5):
	ratings = userElements.index.map(lambda x: userElements.loc[x, 'rating']
	if userElements.loc[x, 'rating'] > 0 else predictRating(userId, userElements.loc[x, 'itemId'], ratingTable,
	                                                        scoreTable, avgRatings, k = k))
	
	return np.array(ratings)


def predictTestRatings(ratingList, ratingTable, scoreTable, avgRating, k = 5):
	predList = ratingList.apply(
		lambda x: predictRating(x['userId'], x['itemId'], ratingTable, scoreTable, avgRating, k), axis = 1)
	
	return predList
