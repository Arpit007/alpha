import pandas as pd
import sys

__DATASET_PATH = 'dataset/tripadvisor/%s.csv'

DATASETS = {
	'Hotels': 'ratings_hotel',
	'Restaurants': 'ratings_restaurants',
	'Attractions': 'ratings_attractions'
}


def getRatingsList(key):
	if key not in DATASETS.keys():
		raise Exception("Invalid Dataset")
	path = __DATASET_PATH % DATASETS[key]
	return pd.read_csv(path, sep = '|')


def getRatingTable(ratingList):
	return pd.pivot_table(data = ratingList, index = 'itemId', columns = 'userId', values = 'rating',
	                      fill_value = 0)


def getItemsDataSet():
	path = __DATASET_PATH % 'items'
	dataset = pd.read_csv(path, sep = '|', index_col = 0)
	return dataset


def getPersonalityDataset():
	path = __DATASET_PATH % 'pers_score'
	dataset = pd.read_csv(path, sep = '|', index_col = 0)
	return dataset


def getIRatingList():
	
	print("Hi, Select from the following Options:")
	keys = list(DATASETS.keys())
	for index, key in enumerate(keys, 1):
		print(index, key)
	
	# Get Item Type
	itemTypeIndex = int(input("Enter Your Choice: "))
	
	if itemTypeIndex == -1:
		sys.exit(0)
	
	if itemTypeIndex < 1 or itemTypeIndex > len(keys):
		raise Exception("Invalid Type")
	
	# Get Rating List
	key = keys[itemTypeIndex - 1]
	ratingList = getRatingsList(key)
	
	return ratingList, key
