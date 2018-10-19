import numpy as np


# Mean Absolute Error
def mae(predVal, trueVal):
	"""Mean Absolute Error"""
	if predVal is None or trueVal is None:
		raise Exception("Invalid Dataset")
	if len(predVal) != len(trueVal):
		raise Exception("Dataset of Unequal Sizes")
	return np.average(np.abs(np.subtract(predVal, trueVal)))


def mse(predVal, trueVal):
	""""Mean Squared Error"""
	if predVal is None or trueVal is None:
		raise Exception("Invalid Dataset")
	if len(predVal) != len(trueVal):
		raise Exception("Dataset of Unequal Sizes")
	return np.average(np.subtract(predVal, trueVal) ** 2)


def rmse(predVal, trueVal):
	""""Root Mean Squared Error"""
	val = mse(predVal, trueVal)
	return np.sqrt(val)


def specificity_precision_recall_accuracy(trueValue, testValue, cutoffThreshold = 3.5):
	"""
	Calculates Specificity, Precision, Recall, Accuracy
	:param trueValue: True Rating Values
	:param testValue: Calculated Values
	:param cutoffThreshold: Rating Threshold
	:return: (specificity, precision, recall, accuracy)
	"""
	truePositive = 0
	trueNegative = 0
	falsePositive = 0
	falseNegative = 0
	
	for (actual, generated) in zip(trueValue, testValue):
		if generated >= cutoffThreshold:
			if actual >= cutoffThreshold:
				truePositive += 1
			else:
				falsePositive += 1
		else:
			if actual >= cutoffThreshold:
				falseNegative += 1
			else:
				trueNegative += 1
	
	recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) != 0 else 0
	specificity = trueNegative / (trueNegative + falsePositive) if (trueNegative + falsePositive) != 0 else 0
	precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) != 0 else 0
	accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative) \
		if (truePositive + trueNegative + falsePositive + falseNegative) != 0 else 0
	
	return [specificity, precision, recall, accuracy]
