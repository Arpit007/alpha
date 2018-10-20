import numpy as np
from src.task import dataset
from src.task import scores
from src.task import rating
from src.utils import splitting
from src.utils import metrics
from src.utils.printer import pprint
import time

"""
SHOULD_MINIMIZE_SET: To Minimize Set
SHOULD_SHUFFLE: To Shuffle the Tes, Train Sets
MINIMUM_USER_RATE_COUNT: Minimum Number of Items to be rated by a user to be in set
MINIMUM_ITEM_RATED_COUNT:Minimum Number of Ratings for an Item to be in set
SUGGESTIONS_COUNT: Count of Suggestions to be given to user
RANDOM_STATE: Set Specific Random State, in case of Shuffling
MULTI_TYPE_TEST: If False, Only test Hotels else all the 3 types
"""

SHOULD_MINIMIZE_SET = True
SHOULD_SHUFFLE = True
MINIMUM_USER_RATE_COUNT = 4
MINIMUM_ITEM_RATED_COUNT = 3
SUGGESTIONS_COUNT = 5
RANDOM_STATE = None
MULTI_TYPE_TEST = True


def run():
	keys = dataset.DATASETS.keys()
	if MULTI_TYPE_TEST is False:
		keys = keys[:1]
	
	for key in keys:
		pprint("Testing for %s" % key, symbolCount = 15, sepCount = 1)
		
		# Load the Datasets
		ratingList = dataset.getRatingsList(key)
		persScoreList = dataset.getPersonalityDataset()
		
		# Minimise dataset for Optimization
		if SHOULD_MINIMIZE_SET is True:
			ratingList, persScoreList = dataset.minimizeSet(ratingList, persScoreList, MINIMUM_ITEM_RATED_COUNT,
			                                                MINIMUM_USER_RATE_COUNT)
		
		testRatingList, trainRatingList = splitting.test_inPlaceTrain_Frame(ratingList, 1, relativeSplit = False,
		                                                                    shuffle = SHOULD_SHUFFLE,
		                                                                    random_state = RANDOM_STATE)
		
		ratingTable = dataset.getRatingTable(trainRatingList)
		sparsity = len(trainRatingList) / np.prod(ratingTable.shape)
		pprint("-> Sparsity: %f%%" % float(sparsity * 100))
		
		# Calculating Pearson Scores
		start_time = time.time()
		pprint('Calculating Pearson Scores')
		
		# Get Users Average Rating
		avgRating = rating.getUsersAverageRating(ratingTable)
		
		pearsonScores = scores.calcAllPearson(ratingTable, avgRating)
		
		# Calculating Personality Scores
		pprint('Calculating Personality Scores')
		personalityScores = scores.calcAllPersonalityScore(ratingTable, persScoreList, avgRating)
		
		# Calculating Pearson Personality Scores
		pprint('Calculating Pearson Personality Scores')
		pearsonPersonalityScores = scores.calcPearsonPersonality(pearsonScores, personalityScores)
		
		pprint("-> Scores Calculated in %.4f seconds" % (time.time() - start_time))
		
		# Calculating Ratings
		pprint('Calculating Ratings & Test Scores')
		start_time = time.time()
		
		testRatingList['pearson'] = rating.predictTestRatings(testRatingList, ratingTable, pearsonScores, avgRating)
		testRatingList['personality'] = rating.predictTestRatings(testRatingList, ratingTable,
		                                                          pearsonPersonalityScores, avgRating)
		
		# Tests
		testLabels = ['Specificity', 'Precision', 'Recall  ', 'Accuracy', 'MAE     ', 'RMSE     ']
		pearsonTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'], testRatingList[
			'pearson'])
		personalityTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'],
		                                                                testRatingList['personality'])
		
		pearsonTest.append(metrics.mae(testRatingList['rating'], testRatingList['pearson']))
		pearsonTest.append(metrics.rmse(testRatingList['rating'], testRatingList['pearson']))
		
		personalityTest.append(metrics.mae(testRatingList['rating'], testRatingList['personality']))
		personalityTest.append(metrics.rmse(testRatingList['rating'], testRatingList['personality']))
		
		pprint("-> Ratings Calculated in %.4f seconds" % (time.time() - start_time))
		
		pprint("Test Scores", symbolCount = 20, sepCount = 1)
		pprint("Method\t\t\t\tPearson\t\t\tPersonality", sepCount = 0, symbolCount = 0)
		for item in zip(testLabels, zip(pearsonTest, personalityTest)):
			print(item[0], "%.4f" % item[1][0], "%.4f" % item[1][1], sep = '\t\t\t')
		
		pprint('', symbolCount = 28, sepCount = 0)
		print("\n\n")
