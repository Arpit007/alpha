def getUserRatedRows(ratingTable, userId):
	
	if not userId in ratingTable.columns:
		raise Exception("Invalid UserId")
	return ratingTable[ratingTable[userId] > 0]


def getUserRatedItems(ratingTable, userId):
	value = getUserRatedRows(ratingTable, userId)
	return value[userId]