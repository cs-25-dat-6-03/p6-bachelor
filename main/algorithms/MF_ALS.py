import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Create full pivot
pivot_table = ratings.pivot(index='userId', columns='movieId', values='rating')
pivot_table = pivot_table.fillna(0)
rating_matrix = pivot_table.to_numpy()

# Create mask for known ratings
mask_full = rating_matrix > 0 
user_item_pairs = np.argwhere(mask_full) # Get all nonzero (user, item) pairs
train_indices, val_indices = train_test_split(user_item_pairs, test_size=0.2, random_state=42) # Split the indices

# Initialize train and val matrices
train_matrix = np.zeros_like(rating_matrix)
val_matrix = np.zeros_like(rating_matrix)

for (u, i) in train_indices:
    train_matrix[u, i] = rating_matrix[u, i]
for (u, i) in val_indices:
    val_matrix[u, i] = rating_matrix[u, i]

# Create masks
train_mask = train_matrix > 0
val_mask = val_matrix > 0

# Alternating Least Squares (ALS) 
def update_U(R, U, V, num_features, lamb, num_users):                                                              # U_i = (V_j * V_j^T + lambda * I)^-1 * R_ij * V_j^T
    for u in range(num_users):
        # Get indices of items user u has rated
        rated_items = R[u, :] > 0                                                                                  # Get the whole u-th row which includes True if user has rated that movie (False otherwise)
        V_rated = V[rated_items]                                                                                   # Selects only the rows of V (movies) where rated_items is True. So if User u has rated movie 1 and 3, then row 1 and 3 are selected
        R_u = R[u, rated_items]                                                                                    # Get all the user u ratings values 

        # Solve for user features (least squares)
        A = V_rated.T @ V_rated + lamb * np.eye(num_features)
        b = V_rated.T @ R_u
        U[u] = np.linalg.solve(A, b)                                                                               # Instead of inverse of matrix (^-1), we use Ax = b linear algorithm (faster and more accurate)

def update_V(R, U, V, num_features, lamb, num_items):                                                              # V_j = (U_i * U_i^T + lambda * I)^-1 * R_ij * U_i^T
    for i in range(num_items):
        # Get indices of users who rated item i
        rated_by = R[:, i] > 0                                                                                     # Get the whole i-th column which includes True if movie has rated by a user (False otherwise)
        U_rated = U[rated_by]                                                                                      # Selects only the rows of U (users) where rated_by is True. So if Movie i has rated by user 1 and 3, then row 1 and 3 are selected
        R_i = R[rated_by, i]                                                                                       # Get the movie i ratings values

        # Solve for item features
        A = U_rated.T @ U_rated + lamb * np.eye(num_features)
        b = U_rated.T @ R_i
        V[i] = np.linalg.solve(A, b)

def compute_rmse(R, U, V):
    prediction = U @ V.T
    non_zero = R > 0
    error = R[non_zero] - prediction[non_zero]
    return np.sqrt(np.mean(error ** 2))

def als(R, num_users, num_items, num_iters = 10, num_features = 4, lamb = 0.1):
    U = np.random.rand(num_users, num_features)
    V = np.random.rand(num_items, num_features)
    for i in range(num_iters):
        update_U(R, U, V, num_features, lamb, num_users)
        update_V(R, U, V, num_features, lamb, num_items)

        rmse = compute_rmse(R, U, V)
        print(f"[ALS] Iteration {i+1}: RMSE = {rmse:.4f}")
    return (U, V)

def predict(U, V):
    return U @ V.T

def compute_mask_rmse(true_R, pred_R, mask):
    error = (true_R - pred_R) * mask
    return np.sqrt(np.sum(error**2) / np.sum(mask))

def hyperparameter_tuning(R, num_iters, lamb, num_features, num_users, num_items):
    best_rmse = float('inf')
    best_params = None

    for rank in num_features:
        for reg in lamb:
            for num_iter in num_iters:
                print(f"Training ALS: features={rank}, lambda={reg}, iterations={num_iter}")

                U, V = als(R, num_users, num_items, num_iter, rank, reg)

                pred = predict(U, V)
                rmse = compute_mask_rmse(val_matrix, pred, val_mask)
                print(f"Validation RMSE = {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (rank, reg, num_iter)
    
                print("Best hyperparameters:", best_params)
                print("Best RMSE:", best_rmse)
    return best_params

def write_to_file(user_id, output_file, unrated_movies_df, predicted_R):
    movie_ratings = ratings.merge(movies, on="movieId")
    user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id]
    with open(filepath + output_file, 'w') as file:
        file.write(f"Movies rated by user {user_id}:\n\n")
        file.write(user_rated_movies[['movieId', 'rating', 'title', 'genres']].to_string())
        file.write(f"\n\n\nTop recommendations for user {user_id}:\n\n")
        file.write(unrated_movies_df.head(10).to_string())
    np.savetxt(filepath + "matrix.txt", np.round(predicted_R, 2), fmt="%.2f", delimiter=",")

def recommend_movies(user_id, R, predicted_R, output, output_file):
    user_index = pivot_table.index.get_loc(user_id)

    # Recommend Movies that user has not watched
    rated_items = R[user_index, :] <= 0 #Select columns and returns true if user has not rated that movie
    unrated_movie_indices = np.where(rated_items)[0]
    unrated_movie_ratings = predicted_R[user_index, unrated_movie_indices]
    unrated_movie_ids = pivot_table.columns[unrated_movie_indices]
    unrated_movie_titles = movies.set_index('movieId').loc[unrated_movie_ids]['title']
    unrated_movie_genres = movies.set_index('movieId').loc[unrated_movie_ids]['genres']

    unrated_movies_df = pd.DataFrame({
        'Movie ID': unrated_movie_ids.values,
        'Predicted Rating': np.round(unrated_movie_ratings,1),
        'Recommended Movies': unrated_movie_titles.values,
        'Genres': unrated_movie_genres.values
    })

    unrated_movies_df = unrated_movies_df.sort_values(by='Predicted Rating', ascending=False)
    print(unrated_movies_df.head(10))  # Show top 10 recommendations

    if output == "y":
        write_to_file(user_id, output_file, unrated_movies_df, predicted_R)

# Print original matrix
num_users, num_items = train_matrix.shape
print(train_matrix)
print(train_matrix.shape, "\n")

# Hyperparameter
num_iters = [10, 50]
num_features = [20, 50, 100]
lamb = [0.01, 0.1]
rank, reg, num_iter = hyperparameter_tuning(train_matrix, num_iters, lamb, num_features, num_users, num_items)
#rank, reg, num_iter = (2, 0.1, 10)

# Predict
U, V = als(train_matrix, num_users, num_items, num_iter, rank, reg)
predicted_R = predict(U, V)
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)

# Recommend
user_id = 1
output = "y"
output_file = "output.txt"
recommend_movies(user_id, train_matrix, predicted_R, output, output_file)