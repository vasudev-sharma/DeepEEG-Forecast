
import numpy as np
import tensorflow

def compute_correlation(true, pred):
    true = true.squeeze()
    pred = pred.squeeze()
    corr_coef = np.corrcoef(true, pred)[0, 1]
    
    return corr_coef 
    
    
def list_correlation(true, pred):
    
    l = []
    time_points = true.shape[-1]
    for i in range(time_points):
        l.append(compute_correlation(true[:, i], pred[:, i]))
    return l

def cosine_loss(y_true, y_pred,axis=1):
    # Compute loss
    y_true = tensorflow.keras.backend.l2_normalize(y_true, axis=axis)
    y_pred = tensorflow.keras.backend.l2_normalize(y_pred, axis=axis)
    return - tensorflow.keras.backend.sum(y_true * y_pred, axis=axis)