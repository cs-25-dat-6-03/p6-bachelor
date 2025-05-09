import numpy as np
import pandas as pd

def write_to_file(user_id, output_file, unrated_movies_df, predicted_R, filepath, ratings, movies):
    movie_ratings = ratings.merge(movies, on="movieId")
    user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id]
    with open(filepath + output_file, 'w') as file:
        file.write(f"Movies rated by user {user_id}:\n\n")
        file.write(user_rated_movies[['movieId', 'rating', 'title', 'genres']].to_string())
        file.write(f"\n\n\nTop recommendations for user {user_id}:\n\n")
        file.write(unrated_movies_df.head(10).to_string())
    np.savetxt(filepath + "matrix.txt", np.round(predicted_R, 2), fmt="%.2f", delimiter=",")

def recommend_movies(user_id, R, predicted_R, output, output_file, pivot_table, filepath, ratings, movies):
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
        write_to_file(user_id, output_file, unrated_movies_df, predicted_R, filepath, ratings, movies)

    return unrated_movies_df.head(10)

def predict(U, V):
    return U @ V.T