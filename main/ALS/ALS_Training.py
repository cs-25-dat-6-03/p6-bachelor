import numpy as np
from ALS import ALS_Evaluation

# # Alternating Least Squares (ALS) 
# def update_U(R, U, V, num_features, lamb, num_users):                                                              # U_i = (V_j * V_j^T + lambda * I)^-1 * R_ij * V_j^T
#     for u in range(num_users):
#         # Get indices of items user u has rated
#         rated_items = R[u, :] > 0                                                                                  # Get the whole u-th row which includes True if user has rated that movie (False otherwise)
#         V_rated = V[rated_items]                                                                                   # Selects only the rows of V (movies) where rated_items is True. So if User u has rated movie 1 and 3, then row 1 and 3 are selected
#         R_u = R[u, rated_items]                                                                                    # Get all the user u ratings values 

#         # Solve for user features (least squares)
#         A = V_rated.T @ V_rated + lamb * np.eye(num_features)
#         b = V_rated.T @ R_u
#         U[u] = np.linalg.solve(A, b)                                                                               # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

# def update_V(R, U, V, num_features, lamb, num_items):                                                              # V_j = (U_i * U_i^T + lambda * I)^-1 * R_ij * U_i^T
#     for i in range(num_items):
#         # Get indices of users who rated item i
#         rated_by = R[:, i] > 0                                                                                     # Get the whole i-th column which includes True if movie has rated by a user (False otherwise)
#         U_rated = U[rated_by]                                                                                      # Selects only the rows of U (users) where rated_by is True. So if Movie i has rated by user 1 and 3, then row 1 and 3 are selected
#         R_i = R[rated_by, i]                                                                                       # Get the movie i ratings values

#         # Solve for item features
#         A = U_rated.T @ U_rated + lamb * np.eye(num_features)
#         b = U_rated.T @ R_i
#         V[i] = np.linalg.solve(A, b)

# def als(R, test_data, num_users, num_items, I, num_iters = 10, num_features = 4, lamb = 0.1):
#     U = 3 * np.random.rand(num_users, num_features)
#     V = 3 * np.random.rand(num_items, num_features)
#     V[0,:] = R[R != 0].mean(axis=0)
#     for i in range(num_iters):
#         update_U(R, U, V, num_features, lamb, num_users)
#         update_V(R, U, V, num_features, lamb, num_items)

#         #rmse = ALS_Evaluation.compute_rmse(test_data, U, V)
#         rmse = ALS_Evaluation.rmse(I, test_data, U, V)
#         print(f"[ALS] Iteration {i+1}: RMSE = {rmse:.4f}")
#     return (U, V)

def update_U(R, U, V, I, lamb, num_features):
    for i in range(I.shape[0]):
        Ii = I.getrow(i).toarray().ravel()
        nui = np.count_nonzero(Ii)  # Number of items user has rated
        if nui == 0: nui = 1

        Ii_nonzero = np.nonzero(Ii)[0]

        V_Ii = V[Ii_nonzero, :]  # (num_rated_items, num_features)
        R_Ii = R[i, Ii_nonzero].toarray().ravel()
        Ai = V_Ii.T @ V_Ii + lamb * nui * np.eye(num_features)
        Vi = V_Ii.T @ R_Ii

        U[i, :] = np.linalg.solve(Ai, Vi)

def update_V(R, U, V, I, lamb, num_features):
    for j in range(I.shape[1]):
        Ij = I.getcol(j).toarray().ravel()
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        if nmj == 0: nmj = 1

        Ij_nonzero = np.nonzero(Ij)[0]

        U_Ij = U[Ij_nonzero, :]  # (num_rated_users, num_features)
        R_Ij = R[Ij_nonzero, j].toarray().ravel()
        Aj = U_Ij.T @ U_Ij + lamb * nmj * np.eye(num_features)
        Vj = U_Ij.T @ R_Ij

        V[j, :] = np.linalg.solve(Aj, Vj)

def als(R, test_data, I, I2, lamb, num_features, num_iters, num_users, num_items):
    U = 3 * np.random.rand(num_users, num_features)
    V = 3 * np.random.rand(num_items, num_features)
    #V[:, 0] = np.array(R.mean(axis=0)).ravel()[:num_items] 
    for k in range(num_iters):
        update_U(R, U, V, I, lamb, num_features)
        update_V(R, U, V, I, lamb, num_features)

        rmse = ALS_Evaluation.rmse(I2, test_data, U, V)
        print(f"[ALS] Iteration {k+1}: RMSE = {rmse:.4f}")
    return (U, V)
