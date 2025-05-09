import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from ALS import ALS_Cold_Start, ALS_Training, ALS_Recommendation, ALS_Hyperparameter

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Create full pivot
pivot_table = ratings.pivot(index='userId', columns='movieId', values='rating')
pivot_table = pivot_table.fillna(0)
rating_matrix = pivot_table.to_numpy()

# Initialize train and val matrices filled with 0s
train_matrix = np.zeros_like(rating_matrix)
val_matrix = np.zeros_like(rating_matrix)

# Create mask for known ratings
mask_full = rating_matrix > 0 
user_item_pairs = np.argwhere(mask_full) # Get all nonzero (in a [user item] format) pairs
train_indices, val_indices = train_test_split(user_item_pairs, test_size=0.2, random_state=42) # Split the indices. 100836 ratings turns into 80668 and 20168 split

# Fill train and validation matrices with ratings based on the splits
for (u, i) in train_indices:
    train_matrix[u, i] = rating_matrix[u, i]
for (u, i) in val_indices:
    val_matrix[u, i] = rating_matrix[u, i]

# Create masks
train_mask = train_matrix > 0
val_mask = val_matrix > 0

# Print original matrix
num_users, num_items = train_matrix.shape
print(train_matrix)
print(train_matrix.shape, "\n")

# User prompts (for cold start problem)
#prompt_user_result = ALS_Cold_Start.prompt_user(filepath, movies, ratings)
#print(prompt_user_result)

# Normalize the training matrix
user_means = np.sum(train_matrix, axis=1) / np.sum(train_mask, axis=1) # Compute user means (ignoring zeros)
user_means = np.nan_to_num(user_means, nan=0.0) # Handle NaN values for users with no ratings
normalized_train_matrix = train_matrix - user_means[:, np.newaxis]
normalized_train_matrix[~train_mask] = 0

# Hyperparameter
num_features = [80, 90, 100]
lamb = [0.001, 0.01, 0.1, 1.0]
num_iters = [20, 50]
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_grid(train_matrix, num_iters, lamb, num_features, num_users, num_items)
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_random(train_matrix, num_users, num_items)
rank, reg, num_iter = (100, 1.0, 50)
print(f"Rank = {rank}, Reg = {reg}, Num_iter = {num_iter}")

# Predict
U, V = ALS_Training.als(train_matrix, num_users, num_items, num_iter, rank, reg)
predicted_R = ALS_Recommendation.predict(U, V)
#predicted_R += user_means[:, np.newaxis] # Add User Means Back to Predictions
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)

# Recommend
user_id = 1
output = True
output_file = "output.txt"
result = ALS_Recommendation.recommend_movies(user_id, train_matrix, predicted_R, output, output_file, pivot_table, filepath, ratings, movies)
print(result)  # Show top 10 recommendations