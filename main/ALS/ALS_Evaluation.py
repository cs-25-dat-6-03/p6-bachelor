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

def rmse(I,R,U,V):
    prediction = ALS_Recommendation.predict(U, V)
    #prediction = np.clip(prediction, 0.5, 5.0)
    return np.sqrt(np.sum((I * (R - prediction))**2)/len(R[R > 0]))

def unexpectedness(i, h):
    cosine = np.dot(i, h) / (np.linalg.norm(i) * np.linalg.norm(h))
    cosine = np.round(cosine,2)
    
    return 1 - (cosine / len(H))

def relevance(i):
    i = min(5.0, max(0.5, i))
    return (i - 0.5)/4.5


def serendipity_eval(i, h, H): # I is the top 100 recommendations of user i, H is the historical interactions of user i  
    return unexpectedness(i, h) * relevance(i)
