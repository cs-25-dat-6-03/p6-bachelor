import pandas as pd
import numpy as np

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

movie_ratings = ratings.merge(movies, on="movieId")
ratings_df = movie_ratings[['userId', 'movieId', 'rating']]

# Determine the number of users and movies
num_users = ratings_df['userId'].max()
num_movies = ratings_df['movieId'].max()

#print(num_users)
#print(num_movies)

# Create the NumPy matrix
rating_matrix = np.zeros((num_users, num_movies))

# Fill the matrix with ratings
for row in ratings_df.itertuples():
    rating_matrix[row.userId - 1, row.movieId - 1] = row.rating

#user = 391
#movie = 2275

#print(rating_matrix)
#print(rating_matrix[user-1,movie-1])

#exit(1)

# Example matrix
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
    [0, 1, 5, 4]
])

# Assigning it to the real matrix instead of example matrix
#R = rating_matrix

