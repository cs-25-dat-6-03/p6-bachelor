import numpy as np
import pandas as pd

# File handling
filepath = "dataset/"
movies = pd.read_csv(filepath + "movies.csv")
ratings = pd.read_csv(filepath + "ratings.csv")

def train_svd(ratings_df, n_factors=20, n_epochs=20, lr=0.005, reg=0.02):
    # Map user and item IDs to indices
    user_ids = ratings_df['userId'].unique()
    item_ids = ratings_df['movieId'].unique()
    
    user2index = {uid: i for i, uid in enumerate(user_ids)}
    item2index = {iid: i for i, iid in enumerate(item_ids)}
    index2user = {i: uid for uid, i in user2index.items()}
    index2item = {i: iid for iid, i in item2index.items()}
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    # Initialize parameters
    global_mean = ratings_df['rating'].mean()
    user_bias = np.zeros(n_users)
    item_bias = np.zeros(n_items)
    P = np.random.normal(scale=0.1, size=(n_users, n_factors))
    Q = np.random.normal(scale=0.1, size=(n_items, n_factors))

    # Convert ratings to index-based triples
    ratings_indexed = [
        (user2index[row.userId], item2index[row.movieId], row.rating)
        for row in ratings_df.itertuples()
    ]
    
    # Training with SGD
    for epoch in range(n_epochs):
        np.random.shuffle(ratings_indexed)
        for u, i, r_ui in ratings_indexed:
            pred = global_mean + user_bias[u] + item_bias[i] + np.dot(P[u], Q[i])
            err = r_ui - pred

            # Update rules
            user_bias[u] += lr * (err - reg * user_bias[u])
            item_bias[i] += lr * (err - reg * item_bias[i])
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P[u] - reg * Q[i])
        
        rmse = compute_rmse(ratings_indexed, global_mean, user_bias, item_bias, P, Q)
        print(f"Epoch {epoch+1}/{n_epochs} - RMSE: {rmse:.4f}")
    
    return {
        'global_mean': global_mean,
        'user_bias': user_bias,
        'item_bias': item_bias,
        'P': P,
        'Q': Q,
        'user2index': user2index,
        'item2index': item2index,
        'index2user': index2user,
        'index2item': index2item
    }

def compute_rmse(ratings_indexed, global_mean, user_bias, item_bias, P, Q):
    errors = []
    for u, i, r_ui in ratings_indexed:
        pred = global_mean + user_bias[u] + item_bias[i] + np.dot(P[u], Q[i])
        errors.append((r_ui - pred) ** 2)
    return np.sqrt(np.mean(errors))

def predict(model, user_id, item_id):
    u_idx = model['user2index'].get(user_id)
    i_idx = model['item2index'].get(item_id)
    if u_idx is None or i_idx is None:
        return model['global_mean']  # fallback for unknown user/item
    pred = (
        model['global_mean'] +
        model['user_bias'][u_idx] +
        model['item_bias'][i_idx] +
        np.dot(model['P'][u_idx], model['Q'][i_idx])
    )
    return pred

def recommend_top_n(model, user_id, n=10):
    u_idx = model['user2index'].get(user_id)
    if u_idx is None:
        return []
    scores = model['global_mean'] + model['item_bias'] + np.dot(model['Q'], model['P'][u_idx])
    item_scores = [(model['index2item'][i], score) for i, score in enumerate(scores)]
    top_n = sorted(item_scores, key=lambda x: x[1], reverse=True)[:n]
    return top_n

model = train_svd(ratings, n_factors=20, n_epochs=25)

print(predict(model, user_id=1, item_id=10))

top_recs = recommend_top_n(model, user_id=1, n=5)
print("Top 5 recommendations for user 1:", top_recs)
