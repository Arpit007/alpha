# Simple Recommender
# A clone of IMDB's Top 250
# 1. Decide on the metric or score to rate movies on.
# 2. Calculate the score for every movie.
# 3. Sort the movies based on the score and output the top results.

import pandas as pd

DATASET_PATH = 'dataset/ml-100k/%s'


def run():
	# Read Dataset
	metadata = pd.read_csv(DATASET_PATH % 'u.data', sep = '\t', names = ['userId', 'itemId', 'rating', 'timestamp'])
	movieList = pd.read_csv(DATASET_PATH % 'u.item', sep = '|', names = ['itemId', 'title'], usecols = [0, 1])
	
	# Calculate Movie Rating Count and Rating Average
	grouper = metadata.groupby('itemId')
	movieData = grouper.size().to_frame(name = 'voteCount')
	movieData = movieData.join(grouper.agg({ 'rating': 'mean' }).rename(columns = { 'rating': 'avgRating' })).reset_index()
	del grouper
	
	# Calculate global Mean Rating, Min. No of Votes(has votes more than 90% of movies)
	globalMeanRating = metadata['rating'].mean()
	minVotes = movieData['voteCount'].quantile(0.9)
	del metadata
	
	# Filter Movie Data on the basis of Min. Votes
	movieData = movieData.loc[movieData['voteCount'] >= minVotes]
	
	# Function that computes the weighted rating of each movie
	def weightedRating(data, m = minVotes, C = globalMeanRating):
		v = data['voteCount']
		R = data['avgRating']
		
		return (v / (v + m) * R) + (m / (m + v) * C)
	
	# Define Score for each movie, Sort on the basis of score
	movieData['score'] = movieData.apply(weightedRating, axis = 1)
	movieData = movieData.sort_values('score', ascending = False)
	
	# Add movie Names
	movieData = pd.merge(movieData, movieList, on = 'itemId')
	del movieList
	
	# Top N movies
	n = int(input('Top N movies: '))
	count = min(n, len(movieData))
	movieData = movieData[['title', 'avgRating', 'score']].head(count)
	
	# Printing Result
	print(movieData)
