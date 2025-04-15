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
    [1, 0, 0, 4],
    [0, 1, 5, 4],
    ])

# Assigning it to the real matrix instead of example matrix
R = rating_matrix

num_users, num_items = R.shape
num_iters = 50 
lamb = 0.1 # Used for overfitting
num_features = 20 

U = np.random.rand(num_users, num_features)
V = np.random.rand(num_items, num_features)

# Alternating Least Squares (ALS)
def update_U(R, U, V):
    for u in range(num_users):
        # Get indices of items user u has rated
        rated_items = R[u, :] > 0
        V_rated = V[rated_items]
        R_u = R[u, rated_items] # Original matrix without the 0s

        # Solve for user features (least squares)
        A = V_rated.T @ V_rated + lamb * np.eye(num_features)
        b = V_rated.T @ R_u
        U[u] = np.linalg.solve(A, b)

def update_V(R, U, V):
    for i in range(num_items):
        # Get indices of users who rated item i
        rated_by = R[:, i] > 0
        U_rated = U[rated_by]
        R_i = R[rated_by, i]

        # Solve for item features
        A = U_rated.T @ U_rated + lamb * np.eye(num_features)
        b = U_rated.T @ R_i
        V[i] = np.linalg.solve(A, b)

def compute_rmse(R, U, V):
    prediction = U @ V.T
    non_zero = R > 0
    error = R[non_zero] - prediction[non_zero]
    return np.sqrt(np.mean(error ** 2))

def als():
    for i in range(num_iters):
        update_U(R, U, V)
        update_V(R, U, V)

        rmse = compute_rmse(R, U, V)
        print(f"Iteration {i+1}: RMSE = {rmse:.4f}")

als()
predicted_R = U @ V.T
print(f"\n{np.round(predicted_R, 2)}")