import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD as TSVD

metadata = pd.read_csv('./dataset/ml-100k/u.data', sep = '\t', names = ['userId', 'itemId', 'rating'],
                       usecols = [0, 1, 2], index_col = 0)
movieList = pd.read_csv('./dataset/ml-100k/u.item', sep = '|', names = ['itemId', 'title'], usecols = [0, 1])

# Rating Matrix
ratingMatrix = metadata.pivot_table(index = 'userId', columns = 'itemId', values = 'rating', aggfunc = np.mean,
                                    fill_value = 0)
del metadata

# Decompose the Matrix
SVD = TSVD(n_components = 12, random_state = 17)
resultantMatrix = SVD.fit_transform(ratingMatrix)

# Pearson r creates a user to user correlation matrix
# We select the user which correlates the most with the user
correlationMatrix = np.corrcoef(resultantMatrix)
del resultantMatrix


def recommend(person, bound):
	person -= 1
	scores = [(correlationMatrix[person, other], other) for other in range(len(correlationMatrix)) if other != person]
	
	scores.sort()
	scores.reverse()
	scores = scores[0:bound]
	
	recommendations = { }
	person += 1
	n = len(ratingMatrix.columns)
	for score, other in scores:
		ranked = ratingMatrix.loc[other + 1]
		for i in range(1, n + 1):
			i1 = ratingMatrix.loc[person, i]
			i2 = ratingMatrix.loc[other + 1, i]
			if i1 == 0 and i2 > 0:
				weight = score * ranked[i]
				
				if i in recommendations:
					s, weights = recommendations[i]
					recommendations[i] = (s + score, weights + [weight])
				else:
					recommendations[i] = (score, [weight])
	
	for r in recommendations:
		sim, item = recommendations[r]
		recommendations[r] = sum(item) / sim
	
	recommendedFrame = pd.DataFrame(list(recommendations.items()), columns = ['itemId', 'score'])
	recommendedFrame = recommendedFrame.sort_values('score', ascending = False)
	
	del scores
	del recommendations
	
	return recommendedFrame


# Get UserId
userId = int(input("Enter User Id: "))
bound = int(input('Enter Bound: '))
n = int(input('Top N Movies: '))
recommendations = recommend(userId, bound)
recommendations = pd.merge(recommendations, movieList, on = 'itemId')

n = min(n, len(recommendations))

print(recommendations['title'].head(n))
