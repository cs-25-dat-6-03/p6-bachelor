import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

#Create User/Item matrix
pivot_table = ratings.pivot(index='userId', columns='movieId', values='rating')
pivot_table = pivot_table.fillna(0)
rating_matrix = pivot_table.to_numpy()

# Normalize ratings
min = ratings["rating"].min()
max = ratings["rating"].max()

# Determine the number of users and movies
num_users = ratings['userId'].unique()
num_movies = ratings['movieId'].unique()
#print(np.sort(num_movies), "\n")

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
print(R)
print(R.shape, "\n")

num_users, num_items = R.shape
num_iters = 50 
lamb = 0.1 # Used for overfitting
num_features = 100

# Create User-Feature and Movie-Feature matrices
U = np.random.rand(num_users, num_features)
V = np.random.rand(num_items, num_features)

# Alternating Least Squares (ALS) 
def update_U(R, U, V):                                                              # U_i = (V_j * V_j^T + lambda * I)^-1 * R_ij * V_j^T
    for u in range(num_users):
        # Get indices of items user u has rated
        rated_items = R[u, :] > 0                                                   # Get the whole u-th row which includes True if user has rated that movie (False otherwise)
        V_rated = V[rated_items]                                                    # Selects only the rows of V (movies) where rated_items is True. So if User u has rated movie 1 and 3, then row 1 and 3 are selected
        R_u = R[u, rated_items]                                                     # Get all the user u ratings values 

        # Solve for user features (least squares)
        A = V_rated.T @ V_rated + lamb * np.eye(num_features)
        b = V_rated.T @ R_u
        U[u] = np.linalg.solve(A, b)                                                # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

def update_V(R, U, V):                                                              # V_j = (U_i * U_i^T + lambda * I)^-1 * R_ij * U_i^T
    for i in range(num_items):
        # Get indices of users who rated item i
        rated_by = R[:, i] > 0                                                      # Get the whole i-th column which includes True if movie has rated by a user (False otherwise)
        U_rated = U[rated_by]                                                       # Selects only the rows of U (users) where rated_by is True. So if Movie i has rated by user 1 and 3, then row 1 and 3 are selected
        R_i = R[rated_by, i]                                                        # Get the movie i ratings values

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
    # Loop until RMSE target is reached
    iteration = 1
    rmse = 1
    rmse_goal = 0.0150
    while rmse > rmse_goal:
    #for i in range(num_iters):
        update_U(R, U, V)
        update_V(R, U, V)

        rmse = compute_rmse(R, U, V)
        print(f"[ALS] Iteration {iteration}: RMSE = {rmse:.4f}")
        iteration+=1
    return U @ V.T

predicted_R = als()
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)

user_id = 3
user_index = pivot_table.index.get_loc(user_id)
#movie_id = 32
#movie_index = pivot_table.columns

# Recommend Movies that user has not watched
rated_items = R[user_index, :] <= 0                                                                 # Select columns and returns true if user has not rated that movie
unrated_movie_indices = np.where(rated_items)[0]
unrated_movie_ratings = predicted_R[user_index, unrated_movie_indices]
unrated_movie_ids = pivot_table.columns[unrated_movie_indices]
unrated_movie_titles = movies.set_index('movieId').loc[unrated_movie_ids]['title']
unrated_movie_genres = movies.set_index('movieId').loc[unrated_movie_ids]['genres']

unrated_movies_df = pd.DataFrame({
    'Movie ID': unrated_movie_ids.values,
    'Predicted Rating': np.round(unrated_movie_ratings,1),
    'Recommended Movies': unrated_movie_titles.values,
    'Genres': unrated_movie_genres.values
    
})

unrated_movies_df = unrated_movies_df.sort_values(by='Predicted Rating', ascending=False)
print(unrated_movies_df.head(10))  # Show top 10 recommendations

# Write to output file
output_file = "output.txt"
movie_ratings = ratings.merge(movies, on="movieId")
user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id]
with open(filepath + output_file, 'w') as file:
    file.write(f"Movies rated by user {user_id}:\n\n")
    file.write(user_rated_movies[['movieId', 'rating', 'title', 'genres']].to_string())
    file.write(f"\n\n\nTop recommendations for user {user_id}:\n\n")
    file.write(unrated_movies_df.head(10).to_string())
np.savetxt(filepath + "matrix.txt", np.round(predicted_R, 2), fmt="%.2f", delimiter=",")