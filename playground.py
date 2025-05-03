import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

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
    

prompt_user()