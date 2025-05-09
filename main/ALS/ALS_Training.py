import numpy as np
from ALS import ALS_Evaluation

# Alternating Least Squares (ALS) 
def update_U(R, U, V, num_features, lamb, num_users):                                                              # U_i = (V_j * V_j^T + lambda * I)^-1 * R_ij * V_j^T
    for u in range(num_users):
        # Get indices of items user u has rated
        rated_items = R[u, :] > 0                                                                                  # Get the whole u-th row which includes True if user has rated that movie (False otherwise)
        V_rated = V[rated_items]                                                                                   # Selects only the rows of V (movies) where rated_items is True. So if User u has rated movie 1 and 3, then row 1 and 3 are selected
        R_u = R[u, rated_items]                                                                                    # Get all the user u ratings values 

        # Solve for user features (least squares)
        A = V_rated.T @ V_rated + lamb * np.eye(num_features)
        b = V_rated.T @ R_u
        U[u] = np.linalg.solve(A, b)                                                                               # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

def update_V(R, U, V, num_features, lamb, num_items):                                                              # V_j = (U_i * U_i^T + lambda * I)^-1 * R_ij * U_i^T
    for i in range(num_items):
        # Get indices of users who rated item i
        rated_by = R[:, i] > 0                                                                                     # Get the whole i-th column which includes True if movie has rated by a user (False otherwise)
        U_rated = U[rated_by]                                                                                      # Selects only the rows of U (users) where rated_by is True. So if Movie i has rated by user 1 and 3, then row 1 and 3 are selected
        R_i = R[rated_by, i]                                                                                       # Get the movie i ratings values

        # Solve for item features
        A = U_rated.T @ U_rated + lamb * np.eye(num_features)
        b = U_rated.T @ R_i
        V[i] = np.linalg.solve(A, b)

def als(R, num_users, num_items, num_iters = 10, num_features = 4, lamb = 0.1):
    U = np.random.rand(num_users, num_features)
    V = np.random.rand(num_items, num_features)
    for i in range(num_iters):
        update_U(R, U, V, num_features, lamb, num_users)
        update_V(R, U, V, num_features, lamb, num_items)

        rmse = ALS_Evaluation.compute_rmse(R, U, V)
        print(f"[ALS] Iteration {i+1}: RMSE = {rmse:.4f}")
    return (U, V)