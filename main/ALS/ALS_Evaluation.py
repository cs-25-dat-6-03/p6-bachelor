import numpy as np
from ALS import ALS_Recommendation

def rmse_b(I, R, U, V, user_index):
    prediction = ALS_Recommendation.predict(U, V, user_index)
    prediction = np.clip(prediction, 0.5, 5.0)
    error = I.multiply(R - prediction)  # Use .multiply for sparse elementwise
    mse = (error.power(2)).sum() / I.sum()
    return np.sqrt(mse)

def rmse(I, R, U, V):
    rows, cols = R.nonzero()
    prediction = np.sum(U[rows] * V[cols], axis=1)
    prediction = np.clip(prediction, 0.5, 5.0)
    error = R[rows, cols].A1 - prediction
    mse = np.mean(error ** 2)
    return np.sqrt(mse)

def rmse_a(I, R, U, V):
    rows, cols = R.nonzero()
    preds = np.sum(U[rows] * V[cols], axis=1)
    preds = np.clip(preds, 0.5, 5.0)
    actuals = R[rows, cols].A1  # .A1 flattens to 1D array
    error = actuals - preds
    mse = np.mean(error ** 2)
    return np.sqrt(mse)

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

def exposure(i,highest,lowest):
    return 100 + ((i - lowest)*(-99)/(highest - lowest))

def exposure_fairness(i_pred, i, highest, lowest):
    return exposure(i, highest, lowest) * relevance(i_pred)