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
	
	rMed = 2.5
	
	if ((r1 > rMed and r2 < rMed) or (r1 < rMed and r2 > rMed)):
		# False Agreement
		distance = 2 * (r1 - r2)
		impact = 1.0 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	else:
		# True Agreement
		distance = abs(r1 - r2)
		impact = ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))
	
	proximity = (2 * (rMax - rMin) + 1 - distance) ** 2
	
	if (r1 > rAvg and r2 > rAvg) or (r1 < rAvg or r2 < rAvg):
		popularity = (1 + ((r1 + r2) / 2 - rAvg) ** 2)
	else:
		popularity = 1
	
	pipScore = proximity * impact * popularity
	return pipScore


def mPip(r1, r2, rAvg, rMin = 1, rMax = 5):
	rMed = 0.5
	r1 = 1 if r1 >= rMed else 0
	r2 = 1 if r2 >= rMed else 0
	rAvg = rAvg / rMax
	
	if ((r1 == 0 and r2 == 1) or (r1 == 1 and r2 == 0)):
		# False Agreement
		distance = 2
		impact = 1.0 / ((abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1))  # Check
	else:
		# True Agreement
		distance = 0
		impact = (abs(r1 - rMed) + 1) * (abs(r2 - rMed) + 1)
	
	proximity = (3 - distance) ** 2
	if (r1 > rAvg and r2 > rAvg) or (r1 < rAvg and r2 < rAvg):
		popularity = 1 + (((r1 + r2) / 2) - rAvg) ** 2
	else:
		popularity = 1
	
	pipScore = proximity * impact * popularity
	return pipScore