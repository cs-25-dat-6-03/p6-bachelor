import pandas as pd
import numpy as np

filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Load the MovieLens 100k data
# Make sure you have the 'u.data' file downloaded from https://grouplens.org/datasets/movielens/100k/
# Load the dataset
# Assuming you have 'u.data' file from the ml-100k dataset in the same folder
df = pd.read_csv(filepath + "ratings.csv")

# Determine the number of users and movies
num_users = df['userId'].max()
num_movies = df['movieId'].max()

print(num_users)
print(num_movies)


# Create the NumPy matrix
rating_matrix = np.zeros((num_users, num_movies))

# Fill the matrix with ratings
for row in df.itertuples():
    rating_matrix[row.userId - 1, row.movieId - 1] = row.rating

user = 391
movie = 2275

# Now rating_matrix[i][j] is the rating user i+1 gave to movie j+1
print(rating_matrix[user-1,movie-1])