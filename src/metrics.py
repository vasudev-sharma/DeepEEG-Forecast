
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

def custom_loss_function(y_true, y_pred,axis=1):
    # Compute loss
    y_true = tenosorflow.backend.keras.l2_normalize(y_true, axis=axis)
    y_pred = tensorflow.backend.keras.l2_normalize(y_pred, axis=axis)
    return - tensorflow.backend.keras.sum(y_true * y_pred, axis=axis)