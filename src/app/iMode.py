import numpy as np
from src.task import dataset
from src.task import users
from src.task import rating
from src.task.algo import Pearson, Personality, Hybrid
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


def run():
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
	
	pearson = Pearson()
	personality = Personality()
	hybrid = Hybrid()
	
	# Calculate Timings of High Computation Tasks
	with Timing() as startTime:
		# Get Users Average Rating
		avgRating = rating.getUsersAverageRating(ratingTable)
		
		# Calculating Pearson Scores
		pprint('Calculating Pearson Scores')
		pearsonScores = pearson.calculate(ratingTable, avgRating)
		
		# Calculating Personality Scores
		pprint('Calculating Personality Scores')
		personality.calculate(ratingTable, avgRating, persScores = persScoreList)
		
		# Calculating Hybrid Scores
		pprint('Calculating Hybrid Scores')
		hybrid.calculate(pearsonScores,avgRating, algo1 = pearson, algo2 = personality, alpha = HYBRID_ALPHA)
		
		pprint("-> Scores Calculated in %.4f seconds" % startTime.getElapsedTime())
	
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
			                                                            hybrid.score, avgRating,
			                                                            k = NEIGHBOURS_COUNT)
			userRatings = userRatings.sort_values('rating', ascending = False)[:SUGGESTIONS_COUNT]
			
			# Suggest Items
			pprint("Suggested %s" % key)
			suggestions = itemsList.loc[userRatings['itemId']]['itemName']
			ratings = userRatings['rating']
			
			for i, item in enumerate(zip(suggestions, ratings), 1):
				if item[1] <= 0:
					break
				print(i, item[0])
