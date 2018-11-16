from src.task import scores
from src.task import rating
from src.utils import metrics
from src.utils.printer import pprint

def calculateScores(task, ratingTable, avgRating, itemAvgRating, persScoreList = None,
                    pearsonScores = None, personalityScores = None, alpha = None):
	pprint('Calculating %s Scores' % task)
	if task == "pearson":
		return scores.calcAllPearson(ratingTable, avgRating)
	elif task == "pip":
		return scores.calcAllPipScores(ratingTable, itemAvgRating)
	elif task == "personality":
		return scores.calcAllPersonalityScore(ratingTable, persScoreList)
	elif task == "hybrid":
		return scores.calcHybrid(pearsonScores, personalityScores, alpha = alpha)
	else:
		raise Exception("Invalid Score Calculator")


def predictAndEvaluate(task, testRatingList, ratingTable, scoreList, avgRating, k):
	if task is None:
		raise Exception("Invalid Task")
	
	testRatingList[task] = rating.predictTestRatings(testRatingList, ratingTable, scoreList, avgRating, k)
	
	testScores = metrics.specificity_precision_recall_accuracy(testRatingList['rating'], testRatingList[task])
	
	testScores.extend([metrics.mae(testRatingList['rating'], testRatingList[task]),
	                   metrics.rmse(testRatingList['rating'], testRatingList[task])])
	return testScores
