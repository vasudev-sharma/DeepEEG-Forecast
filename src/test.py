from utils import plot_multistep_prediction, compare_plot_multistep_prediction

import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":



  
    true = np.load("../models/LR/True.npz")
    pred = np.load("../models/LR/predicted.npz")

    
    true1 = np.load("../models/LSTM/True.npz")
    pred1 = np.load("../models/LSTM/predicted.npz")



    print("LR model")
    print("Shape of the actual test set is - ", true['arr_0'].shape)
    print("Shape of the predicted test set is - ", pred['arr_0'].shape)

    
    print("LSTM model")
    print("Shape of the actual test set is - ", true1['arr_0'].shape)
    print("Shape of the predicted test set is - ", pred1['arr_0'].shape)

    
    compare_plot_multistep_prediction(true['arr_0'], pred['arr_0'], true1['arr_0'], pred1['arr_0'])


    print("hello")
    
 
