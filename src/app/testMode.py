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
MINIMUM_USER_RATE_COUNT: Minimum Number of Items to be rated by a user to be in set
MINIMUM_ITEM_RATED_COUNT:Minimum Number of Ratings for an Item to be in set
SUGGESTIONS_COUNT: Count of Suggestions to be given to user
MULTI_TYPE_TEST: If False, Only test Hotels else all the 3 types
"""

SHOULD_MINIMIZE_SET = True
MINIMUM_USER_RATE_COUNT = 4
MINIMUM_ITEM_RATED_COUNT = 3
SUGGESTIONS_COUNT = 5
MULTI_TYPE_TEST = True


def run():
	keys = dataset.DATASETS.keys()
	if MULTI_TYPE_TEST is False:
		keys = keys[:1]
		
	for key in keys:
		pprint("Testing for %s" % key, symbolCount = 15)
		
		# Load the Datasets
		ratingList = dataset.getRatingsList(key)
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
		
		testRatingList, trainRatingList = splitting.test_inPlaceTrain_Frame(ratingList, 1, False, True)
		
		ratingTable = dataset.getRatingTable(trainRatingList)
		sparsity = len(trainRatingList) / np.prod(ratingTable.shape)
		pprint("Sparsity: %f%%" % float(sparsity * 100))
		
		# Calculating Pearson Scores
		start_time = time.time()
		pprint('Calculating Pearson Scores')
		pearsonScores = scores.calcAllPearson(ratingTable)
		
		# Calculating Personality Scores
		pprint('Calculating Personality Scores')
		personalityScores = scores.calcAllPersonalityScore(ratingTable, persScoreList)
		
		# Calculating Pearson Personality Scores
		pprint('Calculating Pearson Personality Scores')
		pearsonPersonalityScores = scores.calcPearsonPersonality(pearsonScores, personalityScores)
		
		pprint("Scores Calculated in %.4f seconds" % (time.time() - start_time))
		
		# Get Users Average Rating
		usersAvgRating = rating.getUsersAverageRating(ratingTable)
		
		# Calculating Ratings
		pprint('Calculating Ratings & Test Scores')
		start_time = time.time()
		
		testRatingList['pearson'] = rating.predictTestRatings(testRatingList, ratingTable, pearsonScores,
		                                                      usersAvgRating)
		testRatingList['personality'] = rating.predictTestRatings(testRatingList, ratingTable,
		                                                          pearsonPersonalityScores,
		                                                          usersAvgRating)
		
		# Tests
		testLabels = ['Specificity', 'Precision', 'Recall', 'Accuracy', 'MAE', 'RMSE']
		pearsonTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'], testRatingList[
			'pearson'])
		personalityTest = metrics.specificity_precision_recall_accuracy(testRatingList['rating'],
		                                                                testRatingList['personality'])
		
		pearsonTest.append(metrics.mae(testRatingList['rating'], testRatingList['pearson']))
		pearsonTest.append(metrics.rmse(testRatingList['rating'], testRatingList['pearson']))
		
		personalityTest.append(metrics.mae(testRatingList['rating'], testRatingList['personality']))
		personalityTest.append(metrics.rmse(testRatingList['rating'], testRatingList['personality']))
		
		pprint("Ratings Calculated in %.4f seconds" % (time.time() - start_time))
		
		pprint("Test Scores")
		pprint("Method\tPearson\tPersonality", sepCount = 0, symbolCount = 0)
		for item in zip(testLabels, zip(pearsonTest, personalityTest)):
			print("%s\t%.4f\t%.4f" % (item[0], item[1][0], item[1][1]))
		
		print("\n\n\n")
