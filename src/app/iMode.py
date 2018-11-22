import sys

import numpy as np

from src.task import algo, dataset, rating, users
from src.utils.misc import pprint
from src.utils.timing import Timing

"""
SHOULD_MINIMIZE_SET: To Minimize Set
MINIMUM_USER_RATE_COUNT: Minimum Number of Items to be rated by a user to be in set
MINIMUM_ITEM_RATED_COUNT:Minimum Number of Ratings for an Item to be in set
HYBRID_ALPHA: Alpha value to calculate Hybrid Scores
SUGGESTIONS_COUNT: Count of Suggestions to be given to user
"""

SHOULD_MINIMIZE_SET = True
MINIMUM_USER_RATE_COUNT = 4
MINIMUM_ITEM_RATED_COUNT = 3
NEIGHBOURS_COUNT = 5
HYBRID_ALPHA = 0.4
SUGGESTIONS_COUNT = 5


def run(guiMode = False, scores = None):
	# Load the Datasets
	ratingList, key = dataset.getIRatingList()
	persScoreList = dataset.getPersonalityDataset()
	itemsList = dataset.getItemsDataSet()
	
	# Minimise dataset for Optimization
	if SHOULD_MINIMIZE_SET is True:
		ratingList, persScoreList = dataset.minimizeSet(ratingList, persScoreList, MINIMUM_ITEM_RATED_COUNT,
		                                                MINIMUM_USER_RATE_COUNT)
	
	ratingTable = dataset.getRatingTable(ratingList)
	sparsity = 1 - len(ratingList) / np.prod(ratingTable.shape)
	pprint("-> Sparsity: %f%%" % float(sparsity * 100))
	
	userId = 0
	if guiMode is True:
		persScoreList.loc[0] = scores
		ratingTable.insert(0, userId, np.zeros(len(ratingTable)))
	
	personality = algo.Personality()
	pip = algo.Pip()
	hybrid = algo.Hybrid()
	
	# Calculate Timings of High Computation Tasks
	with Timing() as startTime:
		
		# Get Average Ratings
		avgRating = rating.getUsersAverageRating(ratingTable)
		itemsAvgRating = rating.getItemsAverageRating(ratingTable)
		
		# Calculating Personality Scores
		personality.calculate(ratingTable, avgRating, persScores = persScoreList)
		
		# Calculating mPip Scores
		pip.calculate(ratingTable, avgRating, itemsAvgRating = itemsAvgRating)
		
		# Calculating Hybrid Scores
		hybrid.calculate(ratingTable, avgRating, algo1 = personality, algo2 = pip, alpha = HYBRID_ALPHA)
		
		pprint("-> Scores Calculated in %.4f seconds" % startTime.getElapsedTime())
	
	while True:
		
		if guiMode is False:
			# Get userId
			userId = users.getIUserId(ratingTable)
		
		while True:
			# Get City ID
			cityId = users.getICityId(ratingList)
			if cityId == -1:
				if guiMode is True:
					sys.exit(0)
				break
			
			# Calculating Suggestion Ratings
			userRatings = users.getUserItems(userId, cityId, ratingList, ratingTable)
			
			if len(userRatings) != 0:
				personality.predict(ratingTable, avgRating, userRatings, k = NEIGHBOURS_COUNT)
				pip.predict(ratingTable, avgRating, userRatings, k = NEIGHBOURS_COUNT)
				hybrid.predict(ratingTable, avgRating, userRatings, k = NEIGHBOURS_COUNT)
				
				userRatings['rating'] = hybrid.prediction
				userRatings = userRatings.sort_values('rating', ascending = False)[:SUGGESTIONS_COUNT]
				
				# Suggest Items
				pprint("Suggested %s" % key)
				suggestions = itemsList.loc[userRatings['itemId']]['itemName']
				ratings = userRatings['rating']
				
				for i, item in enumerate(zip(suggestions, ratings), 1):
					if item[1] <= 0:
						break
					print(i, item[0])
			else:
				print("No Hotels Found in the city :-(")
		
		guiMode = False
