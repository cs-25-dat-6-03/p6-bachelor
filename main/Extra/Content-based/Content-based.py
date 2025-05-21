import pandas as pd
import numpy as np

# File handling
filepath = "dataset/" 
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

# Merge ratings with movie titles
ratings = pd.merge(ratings, movies, on='movieId')

# Create user-item matrix (users as rows, movies as columns)
user_item_matrix = ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Transpose: movies as rows, users as columns (item-based CF)
item_user_matrix = user_item_matrix.T.values
movie_titles = user_item_matrix.columns.tolist()

# Cosine similarity function using numpy
def cosine_similarity_matrix(A):
    dot_product = np.dot(A, A.T)
    norm = np.linalg.norm(A, axis=1)
    norm_matrix = np.outer(norm, norm)
    similarity = np.divide(dot_product, norm_matrix, out=np.zeros_like(dot_product), where=norm_matrix!=0)
    return similarity

# Compute similarity matrix
similarity_matrix = cosine_similarity_matrix(item_user_matrix)

# Create a mapping of index to movie title
index_to_title = {i: title for i, title in enumerate(movie_titles)}
title_to_index = {title: i for i, title in enumerate(movie_titles)}

# Recommendation function
def get_similar_movies(movie_title, top_n=5):
    if movie_title not in title_to_index:
        return f"Movie '{movie_title}' not found in dataset."
    
    idx = title_to_index[movie_title]
    sim_scores = similarity_matrix[idx]
    
    # Exclude the movie itself
    similar_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
    recommendations = [(index_to_title[i], sim_scores[i]) for i in similar_indices]
    
    return recommendations

# 🔍 Example
movie_name = "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)"
print(f"\nTop Recommendations for {movie_name}:")
for title, score in get_similar_movies(movie_name, top_n=10):
    print(f"{title} (similarity: {score:.4f})")
