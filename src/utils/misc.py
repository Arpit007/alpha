"""
Normalize Scores between -1 and 1
"""
def normalizeScore(scores):
	val = scores.max()
	scores = scores.apply(lambda x: x / val)
	return scores
	
"""
Normalize Scores between -1 and 1
"""
def normalizeScalarScore(scores):
	med = scores.max() / 2
	scores = scores.apply(lambda x: (x - med) / med)
	return scores
