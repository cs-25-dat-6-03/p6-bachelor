import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from ALS import ALS_Cold_Start, ALS_Training, ALS_Recommendation, ALS_Hyperparameter

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Create pivot tables for train and test
train_pivot = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
train_matrix = train_pivot.to_numpy()

# Build lookup for user and item indices (For cold start)
user_ids = list(train_pivot.index) # list of user_ids in matrix
item_ids = list(train_pivot.columns) # list of item_ids in matrix
user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)} # user_id along with its index
item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)} # item_id along with its index

# Extract non-zero ratings and split into train/test
known_ratings = [(u, i, train_matrix[u, i]) 
                 for u in range(train_matrix.shape[0]) 
                 for i in range(train_matrix.shape[1]) 
                 if train_matrix[u, i] > 0]

# Cold start
user_seen = set()
item_seen = set()
guaranteed_train = []

# Guarantee that for every user and item in the test set, at least one of their ratings remains in the training set (Used for cold start)
for u, i, r in known_ratings:
    if u not in user_seen or i not in item_seen:
        guaranteed_train.append((u, i, r))
        user_seen.add(u)
        item_seen.add(i)
remaining_ratings = [x for x in known_ratings if x not in guaranteed_train]

# Split the ratings into 80% training and 20% test data
random.shuffle(remaining_ratings)
split = int(0.8 * len(remaining_ratings))
train_data = guaranteed_train + remaining_ratings[:split]
test_data = remaining_ratings[split:]

# Create a training matrix
R_train = np.zeros_like(train_matrix)
for u, i, r in train_data:
    R_train[u, i] = r

# Print original matrix
num_users, num_items = R_train.shape
print(R_train)
print(R_train.shape, "\n")

# User prompts (for cold start problem)
#prompt_user_result = ALS_Cold_Start.prompt_user(filepath, movies, ratings)
#print(prompt_user_result)

# Hyperparameter
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_grid(R_train, test_data, num_users, num_items)
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_random(R_train, test_data, num_users, num_items)
rank, reg, num_iter = (10, 0.1, 10)
print(f"Rank = {rank}, Reg = {reg}, Num_iter = {num_iter}")

# Predict
U, V = ALS_Training.als(R_train, test_data, num_users, num_items, num_iter, rank, reg)
predicted_R = ALS_Recommendation.predict(U, V)
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)
exit(1)

# Recommend
user_id = 1
output = True
output_file = "output.txt"
result = ALS_Recommendation.recommend_movies(user_id, train_matrix, predicted_R, output, output_file, filepath, ratings, movies)
print(result)  # Show top 10 recommendations