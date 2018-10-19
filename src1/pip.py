def agreement(r1, r2, rMin = 1, rMax = 5):
	rMed = (rMin + rMax) / 2
	return not ((r1 > rMed and r2 < rMed) or (r1 < rMed and r2 > rMed))


def distance(r1, r2, rMin = 1, rMax = 5):
	if agreement(r1, r2, rMin, rMax):
		return abs(r1 - r2)
	else:
		return 2 * abs(r1 - r2)


def proximity(r1, r2, rMin = 1, rMax = 5):
	d = distance(r1, r2, rMin, rMax)
	return (2 * (rMax - rMin) + 1 - d) ** 2


def impact(r1, r2, rMin = 1, rMax = 5):
	rMed = (rMin + rMax) / 2
	if agreement(r1, r2, rMin, rMax):
		return ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	else:
		return 1 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))


def popularity(r1, r2, rAvg):
	if (r1 > rAvg and r2 > rAvg) or (r1 < rAvg or r2 < rAvg):
		return (1 + ((r1 + r2) / 2 - rAvg) ** 2)
	else:
		return 1


def pip(r1, r2, rAvg, rMin = 1, rMax = 5):
	"""
	Combined PIP Function
	r1, r2: Any Two Ratings
	rMax: Max Rating in the Rating Scale (like 5)
	rMin: Min Rating in the Rating Scale (like 1)
	rAvg: Average Rating of Item by All Users
	"""
	if r1 == 0 or r2 == 0:
		return 0
	
	rMed = (rMin + rMax) / 2
	
	if not ((r1 > rMed and r2 < rMed) or (r1 < rMed and r2 > rMed)):
		distance = abs(r1 - r2)
		impact = ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	else:
		distance = 2 * abs(r1 - r2)
		impact = 1 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	
	proximity = (2 * (rMax - rMin) + 1 - distance) ** 2
	
	if (r1 > rAvg and r2 > rAvg) or (r1 < rAvg or r2 < rAvg):
		popularity = (1 + ((r1 + r2) / 2 - rAvg) ** 2)
	else:
		popularity = 1
	
	return proximity * impact * popularity


def similarity(rUsers, avgRating, user1, user2, rMin = 1, rMax = 5):
	"""Calculate Similarity for Items
	rUsers: Ratings by User 1 & 2 & Avg Rating
	avgRating: Avg Rating For Current Item
	user1: User 1 Id
	user2: User 2 Id
	rMax: Max Rating in the Rating Scale (like 5)
	rMin: Min Rating in the Rating Scale (like 1)
	"""
	if user1 == user2:
		return 0
	
	return rUsers.apply(lambda x: pip(x.iloc[0], x.iloc[1], avgRating.iloc[0], rMin, rMax), axis = 1).sum()
