import numpy as np
from ALS import ALS_Recommendation

def compute_rmse(R, U, V):
    prediction = ALS_Recommendation.predict(U, V)
    non_zero = R > 0
    error = R[non_zero] - prediction[non_zero]
    return np.sqrt(np.mean(error ** 2))

def compute_mask_rmse(true_R, pred_R, mask):
    error = (true_R - pred_R) * mask
    return np.sqrt(np.sum(error**2) / np.sum(mask))