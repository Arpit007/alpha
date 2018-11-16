import pandas as pd
import numpy as np


def getUserCount(ratingTable):
	return len(ratingTable.columns)


def getCityCount(ratingList):
	return len(ratingList['cityId'].unique())


def getIUserId(ratingTable):
	userCount = getUserCount(ratingTable)
	userId = int(input("Enter User Id(1-%d): " % userCount))
	
	if (userId < 1 or userId > userCount) and userId != -1:
		raise Exception("Invalid User Id")
	
	return -1 if userId == -1 else ratingTable.columns[userId - 1]


def getUsersFrame(ratingTable):
	users = pd.DataFrame()
	users['userId'] = ratingTable.columns
	return users


def getICityId(ratingList):
	cityCount = getCityCount(ratingList)
	cityId = int(input("Enter City Id(1-%d): " % cityCount))
	if (cityId < 1 or cityId > cityCount) and cityId != -1:
		raise Exception("Invalid City Id")
	return cityId


def getUserItems(userId, cityId, ratingList, ratingTable):
	itemList = list(ratingList[(ratingList['cityId'] == cityId)]['itemId'].unique())
	userElements = ratingTable.loc[itemList][[userId]]
	userElements = userElements.rename(columns = { userId: 'rating' })
	userElements = userElements.reset_index()
	
	return userElements


def getUserRatedRows(ratingTable, userId):
	if not userId in ratingTable.columns:
		raise Exception("Invalid UserId")
	return ratingTable[ratingTable[userId] > 0]


def getUserRatedItems(ratingTable, userId):
	value = getUserRatedRows(ratingTable, userId)
	return value[userId]