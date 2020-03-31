import numpy as np
from tqdm import tqdm 
from metrics import list_correlation

def predict_single_timestep(model, input):

    output = model.predict(input, verbose = 1)
    return output


def predict_multi_timestep(model, input, horizon = 160, model_name = "LR"):
    
    """
    Perform recursive prediction by feeding the network input at time t+1 with the prediction at
    time t. This is repeted 'horizon' number of time.
    :param input: np.array
        (batch_size, window_size, n_features), n_features is supposed to be 1 (univariate time-series)
    :param exogenous: np.array
        exogenous feature for the loads to be predicted
        (batch_size, horizon, n_exog_features)
    :return: np.array
        (batch_size, horizon)
    """

    input_seq = input                                         # (batch_size, n_timestamps, n_features) and (batch_size, n_features * window)

    if model_name == 'LR': 
        output_seq = np.zeros((input_seq.shape[0], horizon, int(input_seq.shape[-1] / 160) ))  # (batch_size, horizon, n_features)
    else: 
        output_seq = np.zeros((input_seq.shape[0], horizon, input_seq.shape[-1]))

    print("Shape of Output Sequence is : - ", output_seq.shape)
    
    for i in tqdm(range(horizon)):

 

        output = predict_single_timestep(model, input_seq)             # [batch_size, n_features] 
    
        input_seq = input_seq.reshape(input_seq.shape[0], horizon, -1 )         #Reshape the size to (Batch_size, horizon, n_features)
        input_seq[:, :-1, :] = input_seq[:, 1:, :]                    

    
        input_seq[:, -1, :] = output


        output_seq[:, i, :] = output

        if model_name == 'LR':
            input_seq = input_seq.reshape(input_seq.shape[0], -1)        #Reshape again so that input is (Batch_size, n_features)

    
    return output_seq


def baseline(test_x, test_y):
    horizon = 160 
    l = []
    print(test_x.shape, test_y.shape)
    for i in tqdm(range( horizon)):
        l.append(np.array(list_correlation(test_x[:,-1, :], test_y[:, i, :])))
    l = np.array(l)
    np.savez_compressed('../models/baseline/baseline_all_channels.npz', l)
    return l

        