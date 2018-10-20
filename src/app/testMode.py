import numpy as np
from src.task import dataset
from src.task import scores
from src.task import rating
from src.utils import splitting
from src.utils import metrics
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
RANDOM_STATE = None
HYBRID_ALPHA = 0.4
MULTI_TYPE_TEST = True


def run():
	keys = dataset.DATASETS.keys()
	if MULTI_TYPE_TEST is False:
		keys = keys[:1]
	
	for key in keys:
		pprint("Testing for %s" % key, symbolCount = 16, sepCount = 1)
		
		# Load the Datasets
		ratingList = dataset.getRatingsList(key)
		persScoreList = dataset.getPersonalityDataset()
		
		# Minimise dataset for Optimization
		if SHOULD_MINIMIZE_SET is True:
			ratingList, persScoreList = dataset.minimizeSet(ratingList, persScoreList, MINIMUM_ITEM_RATED_COUNT,
			                                                MINIMUM_USER_RATE_COUNT)
		
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
			
			# Calculating Pearson Scores
			pprint('Calculating Pearson Scores')
			pearsonScores = scores.calcAllPearson(ratingTable, avgRating)
			
			# Calculating Personality Scores
			pprint('Calculating Personality Scores')
			personalityScores = scores.calcAllPersonalityScore(ratingTable, persScoreList, avgRating)
			
			# Calculating Hybrid Scores
			pprint('Calculating Hybrid Scores')
			hybridScores = scores.calcHybrid(pearsonScores, personalityScores, alpha = HYBRID_ALPHA)
			
			pprint("-> Scores Calculated in %.4f seconds" % startTime.getElapsedTime())
		
		# Calculate Timings of High Computation Tasks
		with Timing() as startTime:
			# Calculating Ratings
			pprint('Calculating Ratings & Test Scores')
			
			testRatingList['pearson'] = rating.predictTestRatings(testRatingList, ratingTable, pearsonScores,
			                                                      avgRating, k = NEIGHBOURS_COUNT)
			testRatingList['personality'] = rating.predictTestRatings(testRatingList, ratingTable, personalityScores,
			                                                          avgRating)
			testRatingList['hybrid'] = rating.predictTestRatings(testRatingList, ratingTable, hybridScores, avgRating)
			
			# Tests
			testLabels = ['Specificity', 'Precision', 'Recall  ', 'Accuracy', 'MAE     ', 'RMSE     ']
			pearsonTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'],
			                                                            testRatingList['pearson'])
			personalityTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'],
			                                                                testRatingList['personality'])
			hybridTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'],
			                                                           testRatingList['hybrid'])
			
			pearsonTest.extend([metrics.mae(testRatingList['rating'], testRatingList['pearson']),
			                    metrics.rmse(testRatingList['rating'], testRatingList['pearson'])])
			
			personalityTest.extend([metrics.mae(testRatingList['rating'], testRatingList['personality']),
			                        metrics.rmse(testRatingList['rating'], testRatingList['personality'])])
			
			hybridTest.extend([metrics.mae(testRatingList['rating'], testRatingList['hybrid']),
			                   metrics.rmse(testRatingList['rating'], testRatingList['hybrid'])])
			
			pprint("-> Ratings Calculated in %.4f seconds" % startTime.getElapsedTime())
		
		pprint("Test Scores", symbolCount = 21, sepCount = 1)
		print("Method\t", "Pearson", "Prsnlty", "Hybrid", sep = "\t\t\t")
		
		for i in range(len(testLabels)):
			print(testLabels[i], "%.4f" % pearsonTest[i], "%.4f" % personalityTest[i], "%.4f" % hybridTest[i],
			      sep = '\t\t\t')
		
		pprint('', symbolCount = 29, sepCount = 0)
		print("\n\n")
