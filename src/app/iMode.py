import numpy as np
from src.task import dataset
from src.task import users
from src.task import scores
from src.task import rating
from src.utils.printer import pprint
import time

"""
SHOULD_MINIMIZE_SET: To Minimize Set
MINIMUM_USER_RATE_COUNT: Minimum Number of Items to be rated by a user to be in set
MINIMUM_ITEM_RATED_COUNT:Minimum Number of Ratings for an Item to be in set
SUGGESTIONS_COUNT: Count of Suggestions to be given to user
"""

SHOULD_MINIMIZE_SET = True
MINIMUM_USER_RATE_COUNT = 4
MINIMUM_ITEM_RATED_COUNT = 3
SUGGESTIONS_COUNT = 5


def run():
	# Load the Datasets
	ratingList, key = dataset.getIRatingList()
	persScoreList = dataset.getPersonalityDataset()
	itemsList = dataset.getItemsDataSet()
	
	# Minimise dataset for Optimization
	if SHOULD_MINIMIZE_SET is True:
		items = ratingList.groupby('itemId').agg({ 'itemId': 'count' }).rename(columns = { 'itemId': 'count' })
		items = items[items['count'] >= MINIMUM_ITEM_RATED_COUNT]
		ratingList = ratingList[ratingList['itemId'].apply(lambda x: x in items.index)]
		
		usersList = ratingList.groupby('userId').agg({ 'userId': 'count' }).rename(columns = { 'userId': 'count' })
		usersList = usersList[usersList['count'] >= MINIMUM_USER_RATE_COUNT]
		ratingList = ratingList[ratingList['userId'].apply(lambda x: x in usersList.index)]
		
		persScoreList = persScoreList.loc[usersList.index]
		del usersList, items
	
	ratingTable = dataset.getRatingTable(ratingList)
	sparsity = len(ratingList) / np.prod(ratingTable.shape)
	pprint("Sparsity: %f%%" % float(sparsity * 100))
	
	# Calculating Pearson Scores
	start_time = time.time()
	pprint('Calculating Pearson Scores')
	
	# Get Users Average Rating
	avgRating = rating.getUsersAverageRating(ratingTable)
	
	pearsonScores = scores.calcAllPearson(ratingTable)
	
	# Calculating Personality Scores
	pprint('Calculating Personality Scores')
	personalityScores = scores.calcAllPersonalityScore(ratingTable, persScoreList, avgRating)
	
	# Calculating Pearson Personality Scores
	pprint('Calculating Pearson Personality Scores')
	pearsonPersonalityScores = scores.calcPearsonPersonality(pearsonScores, personalityScores)
	
	pprint("Scores Calculated in %.4f seconds" % (time.time() - start_time))
	
	while True:
		# Get userId
		userId = users.getIUserId(ratingTable)
		
		if userId == -1:
			break
		
		while True:
			# Get City ID
			cityId = users.getICityId(ratingList)
			if cityId == -1:
				break
			
			# Calculating Suggestion Ratings
			userRatings = users.getUserItems(userId, cityId, ratingList, ratingTable)
			userRatings['rating'] = rating.getOrCalculateUserItemRating(userId, userRatings, ratingTable,
			                                                            pearsonPersonalityScores, avgRating)
			userRatings = userRatings.sort_values('rating', ascending = False)[:SUGGESTIONS_COUNT]
			
			# Suggest Items
			pprint("Suggested %s" % key)
			suggestions = itemsList.loc[userRatings['itemId']]['itemName']
			ratings = userRatings['rating']
			
			for i, item in enumerate(zip(suggestions, ratings), 1):
				if item[1] <= 0:
					break
				print(i, item[0])
