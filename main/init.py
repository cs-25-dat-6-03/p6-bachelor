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

n_users = ratings.userId.unique().shape[0]
n_items = ratings.movieId.unique().shape[0]

# Create mappings from IDs to indices
movie_id_to_index = {mid: idx for idx, mid in enumerate(sorted(ratings.movieId.unique()))}
user_id_to_index = {uid: idx for idx, uid in enumerate(sorted(ratings.userId.unique()))}

# Per-user splitting 
def user_stratified_split(ratings, test_size=0.2, val_size=0.2, seed=42):
    train_rows = []
    val_rows = []
    test_rows = []

    for user_id, user_ratings in ratings.groupby("userId"):
        if len(user_ratings) >= 5: # Users with fewer than 5 ratings go in training (else-statement)
            user_train_val, user_test = train_test_split(user_ratings, test_size=test_size, random_state=seed)
            user_train, user_val = train_test_split(user_train_val, test_size=val_size, random_state=seed)
        else:
            user_train, user_val, user_test = user_ratings, [], []

        train_rows.append(user_train)
        val_rows.append(user_val)
        test_rows.append(user_test)

    train_data = pd.concat(train_rows).reset_index(drop=True)
    val_data = pd.concat(val_rows).reset_index(drop=True)
    test_data = pd.concat(test_rows).reset_index(drop=True)
    return train_data, val_data, test_data

train_data, val_data, test_data = user_stratified_split(ratings)
print(train_data)
exit(1)

# Create training and test matrix
R = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    u = user_id_to_index[line.userId]
    m = movie_id_to_index[line.movieId]
    R[u, m] = line.rating

T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    u = user_id_to_index[line.userId]
    m = movie_id_to_index[line.movieId]
    T[u, m] = line.rating

V = np.zeros((n_users, n_items))
for line in val_data.itertuples():
    u = user_id_to_index[line.userId]
    m = movie_id_to_index[line.movieId]
    V[u, m] = line.rating

# Index matrix for training data
I = R.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# Index matrix for val data
I3 = V.copy()
I3[I3 > 0] = 1
I3[I3 == 0] = 0

# Print original matrix
num_users, num_items = R.shape
print(R)
print(R.shape, "\n")

# User prompts (for cold start problem)
#prompt_user_result = ALS_Cold_Start.prompt_user(filepath, movies, ratings)
#print(prompt_user_result)

# Hyperparameter
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_grid(R, V, num_users, num_items, I3)
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_random(R_train, test_data, num_users, num_items)
rank, reg, num_iter = (90, 0.1, 50)
#print(f"Rank = {rank}, Reg = {reg}, Num_iter = {num_iter}")

# Predict
#U, V = ALS_Training.als(R, T, num_users, num_items, I2, num_iter, rank, reg)
U, V = ALS_Training.als2(R, T, I, I2, reg, rank, num_iter, num_users, num_items)
predicted_R = ALS_Recommendation.predict(U, V)
#predicted_R = np.clip(predicted_R, 0.50, 5.0)
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)
#exit(1)
ALS_Recommendation.save_features(U,V)

# Recommend
user_id = 1
output = True
output_file = "output.txt"
result = ALS_Recommendation.recommend_movies(user_id, R, predicted_R, output, output_file, filepath, ratings, movies)
print(result)  # Show top 10 recommendations