import numpy as np
from src.task import dataset
from src.task import rating
from src.task import algo
from src.utils import splitting
from src.utils.printer import pprint
from src.utils.timing import Timing

"""
SHOULD_MINIMIZE_SET: To Minimize Set
SHOULD_SHUFFLE: To Shuffle the Tes, Train Sets
MINIMUM_USER_RATE_COUNT: Minimum Number of Items to be rated by a user to be in set
MINIMUM_ITEM_RATED_COUNT:Minimum Number of Ratings for an Item to be in set
NEIGHBOURS_COUNT: Similar Neighbours to be taken for prediction
RANDOM_STATE: Set Specific Random State, in case of Shuffling
HYBRID_ALPHA: Alpha value to calculate Hybrid Scores
MULTI_TYPE_TEST: If False, Only test Hotels else all the 3 types
"""

SHOULD_MINIMIZE_SET = True
SHOULD_SHUFFLE = True
MINIMUM_USER_RATE_COUNT = 4
MINIMUM_ITEM_RATED_COUNT = 3
NEIGHBOURS_COUNT = 5
RANDOM_STATE = 173967420
HYBRID_ALPHA = 0.4
MULTI_TYPE_TEST = False


def run():
	keys = list(dataset.DATASETS.keys())
	if MULTI_TYPE_TEST is False:
		keys = keys[:1]
	
	for key in keys:
		pprint("Testing for %s" % key, symbolCount = 16, sepCount = 1)
		
		# Load the Datasets
		ratingList = dataset.getRatingsList(key)
		persScoreList = dataset.getPersonalityDataset()
		
		# Minimise dataset for Optimization
		if SHOULD_MINIMIZE_SET is True:
			ratingList, persScoreList = dataset.minimizeSet(ratingList, persScoreList, MINIMUM_ITEM_RATED_COUNT, MINIMUM_USER_RATE_COUNT)
		
		testRatingList, trainRatingList = splitting.test_inPlaceTrain_split_Frame(ratingList, 1, relativeSplit = False,
		                                                                          shuffle = SHOULD_SHUFFLE,
		                                                                          random_state = RANDOM_STATE)
		
		ratingTable = dataset.getRatingTable(trainRatingList)
		sparsity = 1 - len(trainRatingList) / np.prod(ratingTable.shape)
		pprint("-> Sparsity: %f%%" % float(sparsity * 100))
		
		# Calculate Timings of High Computation Tasks
		with Timing() as startTime:
			# Get Users Average Rating
			avgRating = rating.getUsersAverageRating(ratingTable)
			itemsAvgRating = rating.getItemsAverageRating(ratingTable)
			
			# Calculating Scores
			methods = {
				algo.Pearson.TASK: algo.Pearson(ratingTable, avgRating),
				algo.Pip.TASK: algo.Pip(ratingTable, avgRating, itemsAvgRating = itemsAvgRating),
				algo.MPip.TASK: algo.MPip(ratingTable, avgRating, itemsAvgRating = itemsAvgRating),
				algo.Personality.TASK: algo.Personality(ratingTable, avgRating, persScores = persScoreList)
			}
			methods[algo.Hybrid.TASK] = algo.Hybrid(ratingTable, avgRating, algo1 = methods[algo.Pearson.TASK],
			                                        algo2 = methods[algo.Personality.TASK], alpha = HYBRID_ALPHA)
			
			pprint("-> Scores Calculated in %.4f seconds" % startTime.getElapsedTime())
		
		# Calculate Timings of High Computation Tasks
		with Timing() as startTime:
			
			# Calculating Ratings and Metrics
			for method in methods.values():
				method.predict_evaluate(ratingTable, avgRating, testRatingList, k = NEIGHBOURS_COUNT, itemsAvgRating = itemsAvgRating)
			
			testLabels = ['Specificity', 'Precision', 'Recall  ', 'Accuracy', 'MAE     ', 'RMSE     ']
			
			pprint("-> Ratings Calculated in %.4f seconds" % startTime.getElapsedTime())
		
		pprint("Test Scores", symbolCount = 21, sepCount = 1)
		print("Method\t", *[method.name for method in methods], sep = "\t\t\t")
		
		for i in range(len(testLabels)):
			print(testLabels[i], *["%.4f" % method.metrics[i] for method in methods.values()], sep = '\t\t\t')
		
		pprint('', symbolCount = 29, sepCount = 0)
		print("\n\n")
