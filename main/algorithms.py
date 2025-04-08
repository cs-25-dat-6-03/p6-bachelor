import pandas as pd
import numpy as np

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")



# Original user-item rating matrix (0 = missing rating)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

# Parameters
num_users, num_items = R.shape
k = 2  # number of latent features
alpha = 0.01  # learning rate
beta = 0.02  # regularization parameter
epochs = 5000

# Random initialization of P and Q
P = np.random.rand(num_users, k)
Q = np.random.rand(num_items, k)

# SGD function
def matrix_factorization(R, P, Q, k, steps, alpha, beta):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # Compute error
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    # Update P and Q
                    for r in range(k):
                        P[i][r] += alpha * (2 * eij * Q[r][j] - beta * P[i][r])
                        Q[r][j] += alpha * (2 * eij * P[i][r] - beta * Q[r][j])
    return P, Q.T

# Factorize and reconstruct
nP, nQ = matrix_factorization(R, P, Q, k, epochs, alpha, beta)
reconstructed_R = np.dot(nP, nQ)

# Show predicted ratings
print("Predicted Ratings:\n", np.round(reconstructed_R, 2))

print("hallo")
