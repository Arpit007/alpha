import dask.dataframe as dd
from dask.multiprocessing import get
import multiprocessing as mp
import pandas as pd
import numpy as np
import time

from src1.pip import similarity
from src1.utils.metrics import mae, mse

datasetPath = '../dataset/ml-100k/'

if __name__ == '__main__':
	mp.freeze_support()
	
	metadata = pd.read_csv(datasetPath + 'u1.base', sep = '\t', names = ['userId', 'itemId', 'rating'],
	                       usecols = [0, 1, 2], index_col = 0)
	
	# Generate Rating Table Items*User
	ratingTable = pd.pivot_table(data = metadata, index = 'itemId', columns = 'userId', values = 'rating',
	                             fill_value = 0)
	del metadata
	
	# Generate Average Rating for each Item
	movieRating = pd.DataFrame(np.average(ratingTable, axis = 1), index = ratingTable.index, columns = ['avgRating'])
	
	# Get User Id
	userId = int(input("Enter User Id: "))
	if userId not in ratingTable.columns:
		raise Exception("Invalid User Id")
	
	# Get Similarity Scores
	userScorePath = "save/user%d.p" % userId
	
	# Calculate Similarity Scores in Parallel
	print('Calculating Similarity Scores')
	start_time = time.time()
	
	simTable = pd.DataFrame(data = ratingTable.columns, index = ratingTable.columns)
	simTable = simTable.rename(columns = { 'userId': 'score' })
	
	dSimTable = dd.from_pandas(simTable, npartitions = mp.cpu_count())
	dSimTable['score'] = dSimTable['score'].apply(
		lambda x: similarity(ratingTable[[x, userId]], movieRating['avgRating'], x, userId, 1, 5))
	
	simTable = dSimTable.compute(get = get)
	del dSimTable
	print("Similarity Scores Calculated: %s seconds" % (time.time() - start_time))
	
	# Calculate Ratings in Parallel
	print('Calculating Ratings')
	start_time = time.time()
	
	corRelatedItems = ratingTable[(ratingTable[userId] > 0)]
	corrUserAvgRating = corRelatedItems.apply(lambda x: x[x != 0].mean(), axis = 0)
	

	# Fetch Test Data
	testData = pd.read_csv(datasetPath + 'u1.test', sep = '\t', names = ['userId', 'itemId', 'rating'],
	                       usecols = [0, 1, 2], index_col = 1)
	
	calcRating = testData[testData['userId'] == userId]
	calcRating = pd.DataFrame(calcRating.index, index = calcRating.index)
	calcRating = calcRating.rename(columns = { 'itemId': 'rating' })
	
	dCalcRating = dd.from_pandas(calcRating, npartitions = mp.cpu_count())
	dCalcRating['rating'] = dCalcRating['rating'].apply(lambda x: calcItemRating(x))
	calcRating = dCalcRating.compute(get = get)
	del dCalcRating  # ,corRelatedItems,corrUserAvgRating,simTable
	
	print("Ratings Calculated: %s seconds" % (time.time() - start_time))
	
	
	testRatings = testData[testData['userId'] == userId]['rating']
	result = pd.DataFrame.from_records(
		np.array(testRatings.index.map(lambda x: np.array([testRatings[x], calcRating.at[x, 'rating']]))))
	result = result.fillna(0)
	print('MAE: ', mae(result[0], result[1]))
	print('MSE: ', mse(result[0], result[1]))
	
	
	def calcItemRating(itemId):
		users = ratingTable.loc[itemId]
		users = users[users > 0]
		corrAvg = users.index.map(lambda x: abs(simTable.at[x, 'score']))
		if len(corrAvg) <= 1:
			return 0
		corrAvg = sum(corrAvg) / (len(corrAvg) - 1)
		value = sum(users.index.map(lambda x: simTable.at[x, 'score'] * (users.at[x] - corrUserAvgRating.at[x])))
		return corrUserAvgRating.at[userId] + value / corrAvg
