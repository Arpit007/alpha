import numpy as np
import pandas as pd
from src.utils.misc import pprint
from sklearn.utils import shuffle as shuffleData

MAX_SEED = 2 ** 30


def test_train_split(dataset, testSize = 0.2, relativeSplit = True, shuffle = False, random_state = None, axis = 0):
	"""
	:param dataset: The Dataset to be split
	:param testSize: Size of the test suite
	:param shuffle: Should sets be shuffled
	:param random_state: Random seed for shuffling
	:param axis: 0 for Splitting row-wise, 1 for column-wise
	:return: (testSet, trainSet)
	"""
	if dataset is None:
		raise Exception("Invalid Dataset")
	
	if shuffle is True and random_state is None:
		raise Exception("Provide a Random State")
	
	dataset = dataset if axis == 0 else dataset.T
	
	_dataset = shuffleData(dataset, random_state = random_state) if shuffle is True else dataset
	
	testLength = (int)(len(dataset) * testSize) if relativeSplit else testSize
	
	result = (_dataset.iloc[:testLength].copy(), _dataset.iloc[testLength:].copy())
	
	result = result if axis == 0 else (result[0].T, result[1].T)
	
	return result


def test_train_split_Frame(ratingList, testSize = 0.2, relativeSplit = True, shuffle = False, random_state = None, axis = 0):
	test = []
	train = []
	
	if shuffle is True:
		random_state = (int)(np.random.rand() * MAX_SEED) if random_state is None else random_state
		pprint("-> Random State %d" % random_state)
	
	group = ratingList.groupby('userId')
	for key in group.groups.keys():
		iTest, iTrain = test_train_split(pd.Series(group.groups[key]),
		                                 testSize, relativeSplit, shuffle, random_state, axis)
		test.extend(iTest)
		train.extend(iTrain)
	
	testSet = ratingList.loc[test]
	trainSet = ratingList.loc[train]
	
	return (testSet, trainSet)


def test_inPlaceTrain_split_Frame(ratingList, testSize = 0.2, relativeSplit = True, shuffle = False, random_state = None,
                                  axis = 0):
	test = []
	train = []
	
	if shuffle is True:
		random_state = (int)(np.random.rand() * MAX_SEED) if random_state is None else random_state
		pprint("-> Random State %d" % random_state)
	
	group = ratingList.groupby('userId')
	for key in group.groups.keys():
		iTest, iTrain = test_train_split(pd.Series(group.groups[key]), testSize, relativeSplit, shuffle, random_state,
		                                 axis)
		test.extend(iTest)
		train.extend(iTrain)
	
	testSet = ratingList.loc[test]
	for key in test:
		ratingList.loc[key, 'rating'] = 0
	
	return (testSet, ratingList)
