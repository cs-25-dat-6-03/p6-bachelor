import pandas as pd
import numpy as np

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
movie_ratings = ratings.merge(movies, on="movieId")
output_file = "output.txt"

# Example matrix
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
    ])

# Determine the number of users and movies
num_users = ratings['userId'].max()
num_movies = ratings['movieId'].max()

print(num_users)
print(num_movies)

# Create User Item matrix
rating_matrix = np.zeros((num_users, num_movies))

# Fill the matrix with ratings
for row in ratings.itertuples():
    rating_matrix[row.userId - 1, row.movieId - 1] = row.rating

rating_matrix = rating_matrix[:5,:]
print(rating_matrix)

features = 20 # Features
num_iter = 250 # Iterations
alpha = 0.01

user_matrix = np.random.rand(num_users,features)
movie_matrix = np.random.rand(features,num_movies)

sse = 0
for iterations in range(num_iter):
    old_sse = sse
    sse = 0

    for i in range(5):
        for j in range(10):
            if rating_matrix[i,j] > 0:
                diff = rating_matrix[i,j] - np.dot(user_matrix[i,:], movie_matrix[:,j])
                sse += diff**2

                for k in range(features):
                    user_matrix[i,k] += alpha * (2*diff*movie_matrix[k,j])
                    movie_matrix[k,j] += alpha * (2*diff*user_matrix[i,k])
            #print(j)
        #print(i)
    #print("%d \t\t %f" % (iterations+1, sse))

result = np.round(user_matrix @ movie_matrix,2)
print(result[:5,:])