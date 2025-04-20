from surprise import SVD
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import numpy as np
from surprise import accuracy

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
movie_ratings = ratings.merge(movies, on="movieId")
output_file = "output.txt"

# Sample User-Item rating matrix
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
    [0, 1, 5, 4],
], dtype=float)



# Replace zeros with the row average (basic imputation)
R_filled = R.copy()
row_means = np.true_divide(R.sum(1), (R != 0).sum(1))
for i in range(R.shape[0]):
    R_filled[i, R[i] == 0] = row_means[i]

# SVD decomposition
U, s, VT = np.linalg.svd(R_filled, full_matrices=False)
S = np.diag(s)

# Reduce to k latent features
k = 2
U_k = U[:, :k]
S_k = S[:k, :k]
VT_k = VT[:k, :]

# Reconstruct matrix
R_pred = np.dot(np.dot(U_k, S_k), VT_k)

# Output predicted matrix
print("Predicted Ratings Matrix (NumPy):\n", R_pred)