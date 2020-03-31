from utils import plot_multistep_prediction, compare_plot_multistep_prediction
from input import data
from metrics import list_correlation
from tqdm import tqdm
from predict import baseline
import numpy as np
from numpy import savez_compressed
import matplotlib.pyplot as plt
if __name__ == "__main__":

    '''
    pred = -1
    horizon = 160
    train,  valid, test = data(int(pred), relation= "3", stimulus= "2", horizon = horizon,  split = True , multivariate = True)
    test_X, test_Y = test
    baseline = baseline(test_X, test_Y)
    '''
    
    
    
    

  

    true = np.load("../models/LR/True.npz")
    pred = np.load("../models/LR/predicted.npz")

    
    true1 = np.load("../models/LSTM/True.npz")
    pred1 = np.load("../models/LSTM/predicted.npz")

    baseline = np.load('../models/baseline/baseline_all_channels.npz')



    print("LR model")
    print("Shape of the actual test set is - ", true['arr_0'].shape)
    print("Shape of the predicted test set is - ", pred['arr_0'].shape)

    
    print("LSTM model")
    print("Shape of the actual test set is - ", true1['arr_0'].shape)
    print("Shape of the predicted test set is - ", pred1['arr_0'].shape)

    print("Baseline")
    print("Shape of Baseline model is - ", baseline['arr_0'].shape)
    
    compare_plot_multistep_prediction(true['arr_0'], pred['arr_0'], true1['arr_0'], pred1['arr_0'], baseline['arr_0'])


    print("hello")
 
    
