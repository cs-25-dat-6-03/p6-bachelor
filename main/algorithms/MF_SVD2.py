import numpy as np
import pandas as pd

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
movie_ratings = ratings.merge(movies, on="movieId")

def power_iteration(A, max_iter=100, tol=1e-6):
    v = np.random.rand(A.shape[1])
    
    for _ in range(max_iter):
        Av = np.dot(A, v)  
        norm_av = 0
        for e in Av:
            norm_av += e ** 2
        v_new = Av / np.sqrt(norm_av)

        # Check for convergence
        if np.linalg.norm(v_new - v) < tol:
            break

        v = v_new

    # Compute the corresponding singular value
    eig_val = np.dot(np.dot(A, v), v) / np.dot(v, v)

    return eig_val, v

def svd_with_deflation(A, num_singular_values=1, max_iter=100, tol=1e-6):
    ATA = np.dot(A.T, A)

    n = ATA.shape[0]
    eigen_values = np.zeros(n)
    eigen_vectors = np.zeros((n, n))

    for i in range(num_singular_values):
        # Use power iteration to find the dominant singular vector and value
        singular_value, singular_vector = power_iteration(ATA, max_iter, tol)

        # Store the computed singular vectors and values
        eigen_values[i] = singular_value
        eigen_vectors[:, i] = singular_vector

        # Deflation: Subtract the contribution of the computed singular vector and value
        outer_product = singular_value * np.outer(singular_vector, singular_vector)
        ATA = ATA - outer_product

    # Sort singular values and corresponding vectors in descending order
    sorted_indices = np.argsort(eigen_values)[::-1]
    Sigma = np.array(eigen_values)[sorted_indices]
    Vt = np.array(eigen_vectors)[sorted_indices]

    # Replace zeros in Sigma with a small epsilon to avoid division by zero
    Sigma = np.sqrt(eigen_values)
    Sigma[Sigma == 0] = 1e-10

    # Assemble U, Sigma, and V^T
    U = A.dot(eigen_vectors) / Sigma
    V = eigen_vectors.T

    return U, Sigma, V

user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

#print(user_item_matrix)

A = user_item_matrix.to_numpy()

# Perform SVD with deflation
U, Sigma, Vt = svd_with_deflation(A, num_singular_values=50, max_iter=100, tol=1e-3)

k = 50
U = U[:, :k]
Sigma = Sigma[:k]
Vt = Vt[:k, :]

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# Reconstruct the user-item matrix
reconstructed_matrix = U.dot(np.diag(Sigma)).dot(Vt)

user_id_to_recommend = 3
user_similarity = cosine_similarity(reconstructed_matrix, reconstructed_matrix[user_id_to_recommend])

# Recommend top N movies based on cosine similarity
top_n = 20
recommendations = np.argsort(user_similarity)[::-1][:top_n]

print("Recommended movies for user {}: {}".format(user_id_to_recommend, recommendations))

# Display recommended movies
recommended_movie_ids = user_item_matrix.columns[recommendations]
recommended_movies_info = movies[movies['movieId'].isin(recommended_movie_ids)]
print(recommended_movies_info[['movieId', 'title', 'genres']])


user_rated_movies = movie_ratings[movie_ratings['userId'] == user_id_to_recommend]
output_file = "output.txt"
with open(filepath + output_file, 'w') as file:
    file.write(f"Movies rated by user {user_id_to_recommend}:\n\n")
    file.write(user_rated_movies[['movieId', 'rating', 'title', 'genres']].to_string())
    file.write(f"\n\n\nTop recommendations for user {user_id_to_recommend}:\n\n")
    file.write(recommended_movies_info[['movieId', 'title', 'genres']].to_string())