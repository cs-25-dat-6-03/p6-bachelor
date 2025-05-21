import numpy as np
import pandas as pd
import random

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Create pivot tables for train and test
train_pivot = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
train_matrix = train_pivot.to_numpy()

# Extract non-zero ratings and split into train/test
known_ratings = [(u, i, train_matrix[u, i]) for u in range(train_matrix.shape[0]) for i in range(train_matrix.shape[1]) if train_matrix[u, i] > 0]
#random.shuffle(known_ratings)

split = int(0.8 * len(known_ratings))
train_data = known_ratings[:split]
test_data = known_ratings[split:]

# Create a training matrix
print(train_data[0])
R_train = np.zeros_like(train_pivot)
for u, i, r in train_data:
    print(u)
    print(i)
    print(r)
    exit(1)
    R_train[u, i] = r

# Print original matrix
num_users, num_items = R_train.shape
print(test_data[0])
print(test_data.shape, "\n")
exit(1)

np.random.seed(0)
user_factors = np.random.rand(num_users, num_factors)
item_factors = np.random.rand(num_items, num_factors)

# Optional: you'd normally use ALS to train these matrices, e.g., alternating least squares

# Step 5: Predict function
def predict(u, i):
    return np.dot(user_factors[u], item_factors[i])

# Step 6: Compute RMSE manually
def compute_rmse(test_data):
    se = 0  # sum of squared errors
    for u, i, actual in test_data:
        pred = predict(u, i)
        se += (actual - pred) ** 2
    mse = se / len(test_data)
    return np.sqrt(mse)

# Step 7: Evaluate
rmse = compute_rmse(test_data)
print(f"Test RMSE: {rmse:.4f}")
