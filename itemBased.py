import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD as TSVD

# combined_movies_data = pd.merge(frame, movies, on='item_id')

metadata = pd.read_csv('./dataset/ml-100k/u.data', sep = '\t', names = ['userId', 'itemId', 'rating'],
                       usecols = [0, 1, 2], index_col = 0)
movieList = pd.read_csv('./dataset/ml-100k/u.item', sep = '|', names = ['title'], usecols = [1])

# Rating Matrix
ratingMatrix = metadata.pivot_table(index = 'userId', columns = 'itemId', values = 'rating', aggfunc = np.mean,
                                    fill_value = 0)
del metadata

# Transpose of the Matrix
tRatingMatrix = ratingMatrix.transpose()
del ratingMatrix

# Decompose the Matrix
SVD = TSVD(n_components = 12, random_state = 17)
resultantMatrix = SVD.fit_transform(tRatingMatrix)
del tRatingMatrix

# Pearson r creates a movie to movie correlation matrix
# We select the movie which correlates the most with the movie of interest based on generalized user tastes
correlationMatrix = np.corrcoef(resultantMatrix)
del resultantMatrix

# Get Id of Movie from User
movieId = int(input("Enter Movie Id: ")) - 1
correlationMovie = correlationMatrix[movieId]
recommendationList = [movieList.iloc[id, 0] for id, val in enumerate(correlationMovie) if val < 1.0 and val > 0.9]

for id, movie in enumerate(recommendationList):
	print(id, movie)
