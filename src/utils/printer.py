def pprint(str, sep = '\t', sepCount = 1, symbol = '*', symbolCount = 3):
	"""
	Pretty Print the String
	"""
	fStr = symbol * symbolCount + sep * sepCount + str + sep * sepCount + symbol * symbolCount
	print(fStr)
