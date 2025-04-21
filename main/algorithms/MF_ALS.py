import pandas as pd
import numpy as np

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

user_id = 4
movie_id = 32
user_index = pivot_table.index.get_loc(user_id)
movie_index = pivot_table.columns.get_loc(movie_id)
#print(rating_matrix[user_index, movie_index], "\n")

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

U = np.random.rand(num_users, num_features)
V = np.random.rand(num_items, num_features)

# Alternating Least Squares (ALS) 
def update_U(R, U, V): # U_i = (V_j * V_j^T + lambda * I)^-1 * R_ij * V_j^T
    for u in range(num_users):
        # Get indices of items user u has rated
        rated_items = R[u, :] >= 0.5
        V_rated = V[rated_items]
        R_u = R[u, rated_items] # Original matrix without the 0s

        # Solve for user features (least squares)
        A = V_rated.T @ V_rated + lamb * np.eye(num_features)
        b = V_rated.T @ R_u
        U[u] = np.linalg.solve(A, b) # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

def update_V(R, U, V): # V_j = (U_i * U_i^T + lambda * I)^-1 * R_ij * U_i^T
    for i in range(num_items):
        # Get indices of users who rated item i
        rated_by = R[:, i] >= 0.5
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
        print(f"[ALS] Iteration {i+1}: RMSE = {rmse:.4f}")

als()
predicted_R = U @ V.T
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)

output_file = "output.txt"
#np.savetxt(filepath + output_file, np.round(predicted_R, 2), fmt="%.2f", delimiter=",")