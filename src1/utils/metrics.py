import numpy as np

#Mean Absolute Error
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
	return np.sqrt(np.average(np.subtract(predVal, trueVal) ** 2))