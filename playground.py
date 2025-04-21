import pandas as pd
import numpy as np

# === 1. Load Data ===
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")
ratings_df = ratings.merge(movies, on="movieId")[['userId', 'movieId', 'rating']]

# === 2. Build index mappings ===
user_ids = ratings_df['userId'].unique()
item_ids = ratings_df['movieId'].unique()

user2index = {uid: i for i, uid in enumerate(user_ids)}
item2index = {iid: i for i, iid in enumerate(item_ids)}
index2user = {i: uid for uid, i in user2index.items()}
index2item = {i: iid for iid, i in item2index.items()}

n_users = len(user_ids)
n_items = len(item_ids)

# === 3. Build Rating Matrix for ALS ===
R = np.zeros((n_users, n_items))
for row in ratings_df.itertuples():
    u = user2index[row.userId]
    i = item2index[row.movieId]
    R[u, i] = row.rating

print(R.shape, "\n")
exit(1)

# === 4. ALS Implementation ===
def als(R, num_iters=20, num_features=20, lamb=0.1):
    U = np.random.rand(n_users, num_features)
    V = np.random.rand(n_items, num_features)

    def update_U():
        for u in range(n_users):
            idx = R[u] > 0
            V_r = V[idx]
            R_u = R[u, idx]
            A = V_r.T @ V_r + lamb * np.eye(num_features)
            b = V_r.T @ R_u
            U[u] = np.linalg.solve(A, b)

    def update_V():
        for i in range(n_items):
            idx = R[:, i] > 0
            U_r = U[idx]
            R_i = R[idx, i]
            A = U_r.T @ U_r + lamb * np.eye(num_features)
            b = U_r.T @ R_i
            V[i] = np.linalg.solve(A, b)

    def compute_rmse():
        mask = R > 0
        pred = U @ V.T
        return np.sqrt(np.mean((R[mask] - pred[mask])**2))

    for it in range(num_iters):
        update_U()
        update_V()
        print(f"[ALS] Iteration {it+1}: RMSE = {compute_rmse():.4f}")

    return U, V

U_als, V_als = als(R, num_iters=50)

# === 6. Comparison Functions ===
def predict_als(user_id, movie_id):
    u = user2index.get(user_id)
    i = item2index.get(movie_id)
    if u is not None and i is not None:
        return U_als[u] @ V_als[i]
    return np.nan

# === 7. Example Comparison ===
sample_user = user_ids[2]
sample_movie = item_ids[0]

als_pred = predict_als(sample_user, sample_movie)

print(f"\nPredictions for user {sample_user}, movie {sample_movie}:")
print(f"  ALS prediction: {als_pred:.4f}")
