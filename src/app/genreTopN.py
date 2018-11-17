# Simple Genre Recommender
# A clone of IMDB's Top 250
# 1. Decide on the metric or score to rate movies on.
# 2. Calculate the score for every movie.
# 3. Sort the movies based on the score and output the top results.

import pandas as pd

DATASET_PATH = 'dataset/ml-100k/%s'


def run():
	# Read Dataset
	metadata = pd.read_csv(DATASET_PATH % 'u.data', sep = '\t', names = ['userId', 'itemId', 'rating', 'timestamp'], index_col = 0)
	movieList = pd.read_csv(DATASET_PATH % 'u.item', sep = '|', header = None).rename({ 0: 'itemId', 1: 'title' },
	                                                                                  axis = 'columns')
	genreList = pd.read_csv(DATASET_PATH % 'u.genre', sep = '|', names = ['genre', 'id'])
	
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
	
	# Combine genre
	def combineGenre(data):
		row = movieList[movieList['itemId'] == int(data['itemId'])]
		data['title'] = row.iloc[0, 1]
		genre = []
		for id, ele in enumerate(row.iloc[0, 6:], 1):
			if ele == 1:
				genre.append(id)
		data['genre'] = genre
		return data
	
	# Combine Genres for Movies
	movieData = movieData.apply(combineGenre, axis = 1)
	del movieList
	
	# Print Genre List
	print(genreList[['id', 'genre']].to_string(index = False))
	del genreList
	
	# Get Genre ID
	genre = input('Select Genre(s): ').strip().replace(' ', ',')
	genre = [int(ele) for ele in genre.split(',')]
	movieData = movieData[movieData.apply(lambda x: all(ele in x['genre'] for ele in genre), axis = 1)]
	print('%d Movies Found' % len(movieData))
	del genre
	
	if len(movieData) > 0:
		# Top N movies
		n = int(input('Top N movies: '))
		count = min(n, len(movieData))
		movieData = movieData.head(count).reset_index()[['title', 'avgRating', 'score']]
		
		# Printing Result
		print(movieData)
