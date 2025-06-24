import numpy as np
from ALS import ALS_Evaluation

# # Alternating Least Squares (ALS) 
def update_U(R, U, V, I, lamb, num_features):
    for i in range(I.shape[0]):                                                                         # For each user
        Ii = I.getrow(i).toarray().ravel()                                                              # get a row (a user) in a 1D array
        nui = np.count_nonzero(Ii)                                                                      # Number of items the user has rated
        if nui == 0: nui = 1                                                                            # if user has not rated anything, then assign it 1

        Ii_nonzero = np.nonzero(Ii)[0]                                                                  # Index of all nonzero elements

        V_Ii = V[Ii_nonzero, :]                                                                         # Filter and select data from V based on the index Ii_nonzero
        R_Ii = R[i, Ii_nonzero].toarray().ravel()                                                       # Get data from R on user i, based on index nonzero Ii
        Ai = V_Ii.T @ V_Ii + lamb * nui * np.eye(num_features)                                          # (V^T * V) + (lamb * nui * np.eye(num_features)). Ai is num_features x num_features size. Np.eye creates a square identity matrix of size num_features × num_features
        Vi = V_Ii.T @ R_Ii                                                                              # V^T * R

        U[i, :] = np.linalg.solve(Ai, Vi)                                                               # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

def update_V(R, U, V, I, lamb, num_features):
    for j in range(I.shape[1]):                                                                         # For each movie
        Ij = I.getcol(j).toarray().ravel()                                                              # get a column (a movie) in a 1D array
        nmj = np.count_nonzero(Ij)                                                                      # Number of users that rated movie j
        if nmj == 0: nmj = 1                                                                            # if movie doesnt have any ratings, then assign it 1

        Ij_nonzero = np.nonzero(Ij)[0]                                                                  # Index of all nonzero elements

        U_Ij = U[Ij_nonzero, :]                                                                         # Filter and select data from U based on the index Ij_nonzero
        R_Ij = R[Ij_nonzero, j].toarray().ravel()                                                       # Get data from R on movie j, based on index nonzero Ij
        Aj = U_Ij.T @ U_Ij + lamb * nmj * np.eye(num_features)                                          # (U^T * U) + (lamb * nmi * np.eye(num_features)). Aj is num_features x num_features size. Np.eye creates a square identity matrix of size num_features × num_features
        Vj = U_Ij.T @ R_Ij                                                                              # U^T * R

        V[j, :] = np.linalg.solve(Aj, Vj)                                                               # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

def als(R, test_data, I, I2, lamb, num_features, num_iters, num_users, num_items):
    # Random initialize U and V
    U = 3 * np.random.rand(num_users, num_features)
    V = 3 * np.random.rand(num_items, num_features)

    for k in range(num_iters):
        update_U(R, U, V, I, lamb, num_features)
        update_V(R, U, V, I, lamb, num_features)

        rmse = ALS_Evaluation.rmse(I2, test_data, U, V)
        print(f"[ALS] Iteration {k+1}: RMSE = {rmse:.4f}")
    return (U, V)
