
import numpy as np

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

