import numpy as np
from ALS import ALS_Recommendation

# def compute_rmse(R, U, V):
#     prediction = ALS_Recommendation.predict(U, V)
#     non_zero = R > 0
#     error = R[non_zero] - prediction[non_zero]
#     return np.sqrt(np.mean(error ** 2))

# def compute_rmse(test_data):
#     se = 0  # sum of squared errors
#     for u, i, actual in test_data:
#         pred = predict(u, i)
#         se += (actual - pred) ** 2
#     mse = se / len(test_data)
#     return np.sqrt(mse)

def compute_rmse(test_data, U, V):
    se = 0  # sum of squared errors
    prediction = ALS_Recommendation.predict(U, V)
    for u, i, actual in test_data:
        #pred = prediction[u,i]
        #pred = min(5.0, max(0.5, pred))
        se += (actual - prediction[u,i]) ** 2
    mse = se / len(test_data)
    return np.sqrt(mse)

def rmse(I, R, U, V):
    prediction = ALS_Recommendation.predict(U, V)
    prediction = np.clip(prediction, 0.5, 5.0)
    error = I * (R - prediction)
    return np.sqrt(np.sum(error**2) / np.count_nonzero(R))

def mse(I, R, U, V):
    prediction = ALS_Recommendation.predict(U, V)
    prediction = np.clip(prediction, 0.5, 5.0)
    error = I * (R - prediction)
    return np.sum(error**2) / np.count_nonzero(R)

def unexpectedness(i, H, pivot_table):
    sum = 0
    for k in H:
        h_rating = pivot_table[int(k)]
        h = h_rating.values

        cosine = np.dot(i, h) / (np.linalg.norm(i) * np.linalg.norm(h))
        sum += cosine
    return 1 - (sum / len(H))

def relevance(i):
    if i > 5:
        i = 5
    elif i < 0.5:
        i = 0.5
    return (i - 0.5)/4.5


def serendipity_eval(i, H, pivot_table, i_pred): # I is the top 100 recommendations of user i, H is the historical interactions of user i  
    i_rating = pivot_table[i]
    i = i_rating.values
    return unexpectedness(i, H, pivot_table) * relevance(i_pred)
