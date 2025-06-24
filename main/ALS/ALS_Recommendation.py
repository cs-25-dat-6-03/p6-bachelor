import numpy as np
import pandas as pd
from ALS import ALS_Evaluation

filepath = "dataset/" 

def write_to_file(user_id, output_file, unrated_movies_df, predicted_R, filepath, ratings, movies):
    movie_ratings = ratings.merge(movies, on="movieId")
    user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id]
    Top100s, Top10f = reordering(unrated_movies_df, 'Serendipity', 'Exposure Fairness')
    Top100f, Top10s = reordering(unrated_movies_df, 'Exposure Fairness', 'Serendipity')
    with open(filepath + output_file, 'w') as file:
        file.write(f"Movies rated by user {user_id}:\n\n")
        file.write(user_rated_movies[['movieId', 'rating', 'title', 'genres']].to_string())
        file.write(f"\n\n\nTop recommendations for user {user_id}:\n\n")
        file.write(unrated_movies_df.head(100).to_string())
        file.write(f'\n\n\nSerendipity then Fairness\n\n')
        file.write(f'\n\n\nTop 100 recommendations after first reordering\n\n')
        file.write(Top100s.to_string())
        file.write(f'\n\n\nTop 10 recommendations after second reordering\n\n')
        file.write(Top10f.to_string())
        file.write(f'\n\n\nFairness then Serendipity\n\n')
        file.write(f'\n\n\nTop 100 recommendations after first reordering\n\n')
        file.write(Top100f.to_string())
        file.write(f'\n\n\nTop 10 recommendations after second reordering\n\n')
        file.write(Top10s.to_string())
    np.savetxt(filepath + "matrix.txt", np.round(predicted_R, 2), fmt="%.2f", delimiter=",")

def recommend_movies(user_id, R, T, predicted_R, output, output_file, filepath, ratings, movies):
    # Create full pivot
    pivot_table = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_index = pivot_table.index.get_loc(user_id)

    # Recommend Movies that user has not watched
    user_ratings_dense = R[user_index, :].toarray().ravel()
    rated_items = user_ratings_dense <= 0 #Select columns and returns true if user has not rated that movie
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
    #unrated_movies_df = unrated_movies_df.head(10)
    
    # Serendipty
    user_rated_movies = ratings[ratings['userId'] == user_id]
    Historical = user_rated_movies[['movieId']]

    #unrated_movies_df.head(100)['Movie ID'].values
    I = unrated_movies_df[['Movie ID', 'Predicted Rating']].values
    H = Historical.values
    
    serendipity = []
    fairness = []
    for i,i_pred in I:
        result = ALS_Evaluation.serendipity_eval(i, H, pivot_table, i_pred)
        serendipity.append(result)
        
        #Fairness 
        nonzero_ratings = pivot_table[i][pivot_table[i] > 0]
        i_exposure = nonzero_ratings.shape[0]

        nonzero_counts = (pivot_table > 0).sum(axis=0)
        highest = nonzero_counts.max()
        lowest = nonzero_counts.min()
        result = ALS_Evaluation.exposure_fairness(i_pred, i_exposure, highest, lowest) 
        fairness.append(result)

    unrated_movies_df['Serendipity'] = np.round(serendipity,2)
    unrated_movies_df['Exposure Fairness'] = np.round(fairness,2)

    # Test data
    # test_movie_indices = T[user_index, :].nonzero()[1]  # Get nonzero column indices for this user in T (movies in test set)
    # test_movie_ids = pivot_table.columns[test_movie_indices]  # Map indices to movie IDs
    # test_ratings = T[user_index, test_movie_indices].toarray().ravel() # Get the ratings from T for these indices

    # # Filter recommendations for those in the test set
    # mask = unrated_movies_df['Movie ID'].isin(test_movie_ids)
    # unrated_movies_df = unrated_movies_df[mask]

    if output:
        write_to_file(user_id, output_file, unrated_movies_df, predicted_R, filepath, ratings, movies)

    return unrated_movies_df.head(100)
    
def reordering(Ranking_list, Reordering1, Reordering2):

    #unrated_movies_df = unrated_movies_df.sort_values(
    #by=['Serendipty'],
    #ascending=[False])

    Ranking_list1 = Ranking_list.sort_values(
    by=[Reordering1],
    ascending=[False])

    Ranking_list1 = Ranking_list1.head(100)
    Ranking_list2 = Ranking_list1.sort_values(
    by=[Reordering2],
    ascending=[False])

    return (Ranking_list1, Ranking_list2.head(10)) 

def save_features(U, V):
    np.savetxt(filepath + "user_matrix.txt", np.round(U, 2), fmt="%.2f", delimiter=",")
    np.savetxt(filepath + "item_matrix.txt", np.round(V, 2), fmt="%.2f", delimiter=",")

def predict(U, V, user_index):
    return U @ V.T