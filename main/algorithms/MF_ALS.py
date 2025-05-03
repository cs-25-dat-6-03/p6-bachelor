import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

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

def predict(U, V):
    return U @ V.T

def compute_rmse(R, U, V):
    prediction = predict(U, V)
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

def hyperparameter_tuning_random(R, num_users, num_items):
    # Define search space
    latent_feature_choices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    iteration_choices = [5, 10, 20, 30, 40, 50]
    lambda_choices = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    # How many random combinations to try
    max_trials = 60
    early_stopping_rounds = 10
    no_improve_count = 0

    best_rmse = float('inf')
    best_params = None

    for trial in range(max_trials):
        # Randomly sample parameters
        rank = random.choice(latent_feature_choices)
        num_iter = random.choice(iteration_choices)
        reg = random.choice(lambda_choices)
        print(f"Training ALS: features={rank}, lambda={reg}, iterations={num_iter}")

        U, V = als(R, num_users, num_items, num_iter, rank, reg)

        pred = predict(U, V)
        rmse = compute_mask_rmse(val_matrix, pred, val_mask)
        print(f"Trial = {trial+1}, Validation RMSE = {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (rank, reg, num_iter)
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}")

        # Early stopping
        if no_improve_count >= early_stopping_rounds:
            print(f"\nEarly stopping after {trial+1} number of trials with no improvement for {early_stopping_rounds} rounds.")
            break

        print("Best hyperparameters:", best_params)
        print(f"Best RMSE: {best_rmse:.4f}" )
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

    if output:
        write_to_file(user_id, output_file, unrated_movies_df, predicted_R)

    return unrated_movies_df.head(10)

def prompt_user(): # For cold start
    print("\nHello! Which genres are you interested in?")
    all_genres = movies['genres'].str.split('|').explode().unique()[:-1] # Show all genres without the last one (because it is a no genre useless string)

    # Display the list of genres
    genres_array = list(all_genres)
    print("Available genres:")
    for i, genre in enumerate(genres_array, start=1):
        print(f"{i}. {genre}")
    
    # Genres User input
    genres_choice = input("\nWhich genres are you interested in? (separate input with a comma. For instance: 1, 3, 6)\n")
    genres_choice_array = []
    for choice in genres_choice.split(','):
        genres_choice_array.append(genres_array[int(choice)-1])
    print("\nYou selected the following genres:", genres_choice_array)

    # Precompute global averages
    avg_ratings = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
    C = avg_ratings['count'].mean()
    m = avg_ratings['mean'].mean()

    max_user_id = ratings['userId'].max()
    selected_movie_ids = set()  # Keep track of already selected movies

    for genre in genres_choice_array:
        # Filter movies by genre
        movies_list = movies[movies['genres'].str.contains(genre, na=False)]

        # Calculate average ratings for the filtered movies
        def bayesian_avg(ratings):
            return round((C * m + ratings.sum()) / (C + ratings.count()), 2)

        bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
        bayesian_avg_ratings.columns = ['movieId', 'avg_rating']

        # Merge with movie details
        movies_list = movies_list.merge(bayesian_avg_ratings, left_on='movieId', right_on='movieId', how='left')
        movies_list = movies_list.sort_values(by='avg_rating', ascending=False)

        # Exclude already selected movies
        movies_list = movies_list[~movies_list['movieId'].isin(selected_movie_ids)] # "~" = NOT Operator

        # Display the top 10 movies
        print(f"\nTop 10 {genre} movies based on average rating:")
        movies_list = movies_list.head(10).reset_index(drop=True)
        movies_list.index += 1 
        print(movies_list[['title', 'genres', 'avg_rating']].to_string(index=True))

        # Add selected movies to the set
        selected_movie_ids.update(movies_list['movieId'].tolist())

        # Prepare the list of movie IDs for user selection
        movies_array = movies_list['movieId'].tolist()

        # Movies User input
        movies_choice = input(f"\nWhich {genre} movies are you interested in? (separate input with a comma. For instance: 1, 3, 6)\n")
        for choice in movies_choice.split(','):
            new_rating = {
                'userId': max_user_id + 1,
                'movieId': movies_array[int(choice)-1],
                'rating': 5.0,
                'timestamp': 1683033600  # Example timestamp (Unix time)
            }
            new_rating_df = pd.DataFrame([new_rating])
            new_rating_df.to_csv(filepath + "ratings.csv", mode='a', index=False, header=False) # Opens the file in append mode
    return "\nYou have been added to database and can now get recommendations based on your selected movies!\n"

# Print original matrix
num_users, num_items = train_matrix.shape
print(train_matrix)
print(train_matrix.shape, "\n")

# User prompts (for cold start problem)
prompt_user_result = prompt_user()
print(prompt_user_result)

# Hyperparameter
num_iters = [10, 50]
num_features = [20, 50, 100]
lamb = [0.01, 0.1]
#rank, reg, num_iter = hyperparameter_tuning(val_matrix, num_iters, lamb, num_features, num_users, num_items)
#rank, reg, num_iter = hyperparameter_tuning_random(val_matrix, num_users, num_items)
rank, reg, num_iter = (80, 0.001, 50)
print(f"Rank = {rank}, Reg = {reg}, Num_iter = {num_iter}")

# Predict
U, V = als(train_matrix, num_users, num_items, num_iter, rank, reg)
predicted_R = predict(U, V)
print(f"\n{np.round(predicted_R, 2)}")
print(predicted_R.shape)

# Recommend
user_id = 1
output = True
output_file = "output.txt"
result = recommend_movies(user_id, train_matrix, predicted_R, output, output_file)
print(result)  # Show top 10 recommendations