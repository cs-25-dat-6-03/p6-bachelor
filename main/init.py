import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from ALS import ALS_Cold_Start, ALS_Training, ALS_Recommendation, ALS_Hyperparameter, ALS_Evaluation

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

n_users = ratings.userId.unique().shape[0] # 610
n_items = ratings.movieId.unique().shape[0] # 9724

# Create mappings from IDs to indices
movie_id_to_index = {mid: idx for idx, mid in enumerate(sorted(ratings.movieId.unique()))} # mid = movieId, idx = index/counter for each movie
user_id_to_index = {uid: idx for idx, uid in enumerate(sorted(ratings.userId.unique()))} # uid = userId, idx = index/counter for each user

# Per-user splitting 
def user_split(ratings, test_size=0.2, val_size=0.2, seed=42):
    train_rows = []
    val_rows = []
    test_rows = []

    for user_id, user_ratings in ratings.groupby("userId"):
        if len(user_ratings) >= 5: 
            user_train_val, user_test = train_test_split(user_ratings, test_size=test_size, random_state=seed) # total 100837, validation 80669, test 20168
            user_train, user_val = train_test_split(user_train_val, test_size=val_size, random_state=seed) # total 80669, training 64535, validation 16134
        else: 
            # Users with fewer than 5 ratings go in training and doesnt get split
            user_train, user_val, user_test = user_ratings, [], [] 

        train_rows.append(user_train)
        val_rows.append(user_val)
        test_rows.append(user_test)

    # Combines the rows into a dataset
    train_data = pd.concat(train_rows).reset_index(drop=True) 
    val_data = pd.concat(val_rows).reset_index(drop=True)
    test_data = pd.concat(test_rows).reset_index(drop=True)
    return train_data, val_data, test_data

train_data, val_data, test_data = user_split(ratings)

# Create training and test matrices
def df_to_sparse_matrix(df, n_users, n_items, user_id_to_index, movie_id_to_index):
    row_inds = df['userId'].map(user_id_to_index)
    col_inds = df['movieId'].map(movie_id_to_index)
    data = df['rating']
    return csr_matrix((data, (row_inds, col_inds)), shape=(n_users, n_items))

# Create Sparse rating matrices (It stores the positions of the nonzero values)
R = df_to_sparse_matrix(train_data, n_users, n_items, user_id_to_index, movie_id_to_index)
T = df_to_sparse_matrix(test_data, n_users, n_items, user_id_to_index, movie_id_to_index)
V = df_to_sparse_matrix(val_data, n_users, n_items, user_id_to_index, movie_id_to_index)

# index matrices (used for eval and mask)
I = R.copy()
I.data = np.ones_like(I.data) # Replaces known values with "1"

I2 = T.copy()
I2.data = np.ones_like(I2.data)

I3 = V.copy()
I3.data = np.ones_like(I3.data)

# Print original matrix
num_users, num_items = R.shape
#print(R)
print(R.shape, "\n")

# User prompts (for cold start problem)
#prompt_user_result = ALS_Cold_Start.prompt_user(filepath, movies, ratings)
#print(prompt_user_result)

# Hyperparameter
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_grid(R, V, num_users, num_items, I, I3)
#rank, reg, num_iter = ALS_Hyperparameter.hyperparameter_tuning_random(R_train, test_data, num_users, num_items)
rank, reg, num_iter = (100, 0.1, 50)
print(f"Rank = {rank}, Reg = {reg}, Num_iter = {num_iter}")

# Predict
user_id = 1
#U, V = ALS_Training.als(R, T, num_users, num_items, I2, num_iter, rank, reg)
U, V = ALS_Training.als(R, T, I, reg, rank, num_iter, num_users, num_items)
predicted_R = ALS_Recommendation.predict(U, V, user_id_to_index[user_id])
#predicted_R = np.clip(predicted_R, 0.5, 5.0)
print(f"\n{np.round(predicted_R, 2)}")
#print(f"\n{np.round(U @ V.T, 2)}")
print(predicted_R.shape)
ALS_Recommendation.save_features(U,V)
exit(1)

# Recommend

output = True
output_file = "output.txt"
result = ALS_Recommendation.recommend_movies(user_id, R, T, predicted_R, output, output_file, filepath, ratings, movies)
print(result)  # Show top 10 recommendations