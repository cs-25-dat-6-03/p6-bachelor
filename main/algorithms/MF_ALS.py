import pandas as pd
import numpy as np

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Alternating Least Squares (ALS)

R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

num_users, num_items = R.shape
num_iters = 50
lamb = 0.1
num_features = 2 

U = np.random.rand(num_users, num_features)
V = np.random.rand(num_items, num_features)

def update_U():
    for u in range(num_users):
        # Get indices of items user u has rated
        rated_items = R[u, :] > 0
        V_rated = V[rated_items]
        R_u = R[u, rated_items] # Original matrix without the 0s

        # Solve for user features (least squares)
        A = V_rated.T @ V_rated + lamb * np.eye(num_features)
        b = V_rated.T @ R_u
        U[u] = np.linalg.solve(A, b)

def update_V():
    for i in range(num_items):
        # Get indices of users who rated item i
        rated_by = R[:, i] > 0
        U_rated = U[rated_by]
        R_i = R[rated_by, i]

        # Solve for item features
        A = U_rated.T @ U_rated + lamb * np.eye(num_features)
        b = U_rated.T @ R_i
        V[i] = np.linalg.solve(A, b)

def als():
    for i in range(num_iters):
        update_U()
        update_V()

als()

predicted_R = U @ V.T

print(predicted_R)