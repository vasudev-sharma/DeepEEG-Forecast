
import numpy as np

def compute_correlation(true, pred):
    true = true.squeeze()
    pred = pred.squeeze()
    corr_coef = np.corrcoef(true, pred)[0, 1]
    
    return corr_coef 
    
    
def list_correlation(electi, true, pred):
    l = []
    for i in range(len(electi)):
        l.append(compute_correlation(true[:, i], pred[:, i]))

