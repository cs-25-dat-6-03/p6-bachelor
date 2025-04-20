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

def FunkSVD():
    return 0

R = np.array(R)
N = len(R)
M = len(R[0])
K = 20 # Features
num_iter = 250

P = np.random.rand(N,K)
Q = np.random.rand(M,K)
