def normalizeScore(scores):
	"""
	Normalize Scores
	"""
	val = scores.max()
	if val > 1.0:
		scores = scores.apply(lambda x: x / val)
	return scores


def pprint(str, sep = '\t', sepCount = 0, symbol = '*', symbolCount = 0):
	"""
	Pretty Print the String
	"""
	fStr = symbol * symbolCount + sep * sepCount + str + sep * sepCount + symbol * symbolCount
	print(fStr)


COLUMN_LENGTH = 15


def getRowFormat(length):
	columnFormat = "{:>%d}" % COLUMN_LENGTH
	rowFormat = columnFormat * length
	
	return rowFormat
