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


def predict_autoencoder(encoder_model, decoder_model, encoder_inputs):
    
    """
    Multi step Inference (1 at a time)
    :param encoder_inputs: numpy.array
        Encoder input: shape(n_samples, input_sequnece_length, n_features)

 
    :return: numpy.array
        shape(n_samples, output_sequence_length, features)
    """

    horizon = encoder_inputs.shape[1]
    features = encoder_inputs.shape[-1]

    # predictions, shape (batch_size, horizon, 1)
    predictions = np.zeros((encoder_inputs.shape[0], horizon, features))

    # produce embeddings with encoder
    states_value = encoder_model.predict(encoder_inputs)  # [h,c](lstm) or [h](gru) each of dim (batch_size, n_hidden)

    # populate the decoder input with the last encoder input
    decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))  # decoder input for a single timestep
    decoder_input[:, 0, 0] = encoder_inputs[:, -1, 0]

    for i in tqdm(range(horizon)):
        

        if isinstance(states_value, list):
            outputs = decoder_model.predict([decoder_input] + states_value)
        else:
            outputs = decoder_model.predict([decoder_input, states_value])

        # prediction at timestep i
        output = outputs[0]  # output (batch_size, 1, 1)
        predictions[:, i, 0] = output[:, 0, 0]

        # Update the decoder input with the predicted value (of length 1).
        decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))
        decoder_input[:, 0, 0] = output[:, 0, 0]

        # Update states
        states_value = outputs[1:] # h, c (both [batch_size, n_hidden]) or just h

    return predictions




def baseline(test_x, test_y):
    horizon = 160 
    l = []
    print(test_x.shape, test_y.shape)
    for i in tqdm(range( horizon)):
        l.append(np.array(list_correlation(test_x[:,-1, :], test_y[:, i, :])))
    l = np.array(l)
    np.savez_compressed('../models/baseline/baseline_all_channels.npz', l)
    return l

