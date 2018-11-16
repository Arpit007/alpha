"""
Normalize Scores between -1 and 1
"""
def normalizeScore(scores):
	val = scores.max()
	scores = scores.apply(lambda x: x / val)
	return scores

def pprint(str, sep = '\t', sepCount = 0, symbol = '*', symbolCount = 0):
	"""
	Pretty Print the String
	"""
	fStr = symbol * symbolCount + sep * sepCount + str + sep * sepCount + symbol * symbolCount
	print(fStr)
