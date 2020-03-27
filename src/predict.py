import numpy as np
from tqdm import tqdm 

def predict_single_timestep(model, input):

    output = model.predict(input, verbose = 1)
    return output



def predict_multi_timestep(model, input, horizon):
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
    input_seq = input                                         # (batch_size, n_timestamps, n_features)
    output_seq = np.zeros((input_seq.shape[0], horizon, input_seq.shape[-1] ))  # (batch_size, horizon, n_features)
    for i in tqdm(range(horizon)):
    
        output = predict_single_timestep(model, input_seq)             # [batch_size, n_features]
        print(output.shape)
        input_seq[:, :-1, :] = input_seq[:, 1:, :]                    
        print(input_seq.shape)
        print(output_seq.shape)
        input_seq[:, -1, :] = output

        '''if exogenous is not None:
            input_seq[:, -1, 1:] = exogenous[:, i, :]
        '''
        # input_seq = np.concatenate([input_seq[:, 1:, :], np.expand_dims(output,axis=-1)], axis=1)
        output_seq[:, i, :] = output
    return output_seq

