
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
import tensorflow.keras
import tensorflow
from metrics import cosine_loss

'''Linear Regression Models'''

def linear_regression(dim, learning_rate):

    _, features = dim 
    out_features = features / 160

  

    model = Sequential([
    Dense( int(out_features), input_shape = (features,) ,activation = "linear" )])

     #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])

    return model 



'''CNN Models'''

#COnvolutional Neural Network
def conv_1D(dim, source_Y, learning_rate):

    _, window, features = dim

    inp = Input([window, features])
    X = inp
    
    X = Conv1D(filters = 16, kernel_size = 5)(X)
    X= BatchNormalization()(X)
    X = ELU()(X)
    X= Dropout(0.1)(X)
    X = MaxPooling1D(pool_size= 2)(X)

  
  
    X = Conv1D(filters = 40, kernel_size = 3)(X)
    X= BatchNormalization()(X)
    X = ELU()(X)
    X= Dropout(0.1)(X)
    X = MaxPooling1D(pool_size= 2)(X)


    X = Conv1D(filters = 40, kernel_size = 3)(X)
    X= BatchNormalization()(X)
    X = ELU()(X)
    X= Dropout(0.1)(X)
    X = MaxPooling1D(pool_size= 2)(X)
    
    X = Flatten()(X)
    #out = Activation("linear")(X)
    #X = Dense(50, activation = "relu")(X)
    out = Dense(source_Y, activation = "linear",  kernel_initializer = 'normal')(X)

     #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
    
    model = Model(inputs = inp, outputs = out)
    return model



'''
Models for Tuning
'''


def conv_1D_hp(hp):

  window = 160 
  learning_rate = 0.01
  features = 64
 
  model = Sequential([

    Conv1D(input_shape = (window, features), filters = hp.Int('conv_1_filter', min_value=8, max_value= 64, step= 8),  kernel_size=hp.Choice('conv_1_kernel', values = [3,5])),
    LeakyReLU(),
    MaxPooling1D(pool_size= 2),

  
    Conv1D(filters = hp.Int('conv_2_filter', min_value= 8, max_value=64, step= 8 ), kernel_size = hp.Choice('conv_2_kernel', values = [3,5])),
      LeakyReLU(),
      MaxPooling1D(pool_size= 2),

      
      Conv1D(filters =  hp.Int('conv_3_filter', min_value= 8, max_value=64, step= 8 ), kernel_size = hp.Choice('conv_3_kernel', values = [3,5])),
      LeakyReLU(),
      MaxPooling1D(pool_size= 2),

      
      Conv1D(filters =  hp.Int('conv_4_filter', min_value= 8, max_value=64, step= 8 ), kernel_size = hp.Choice('conv_4_kernel', values = [3,5])),
      LeakyReLU(),
      MaxPooling1D(pool_size= 2),
  
      
      Flatten(),
      #out = Activation("linear")(X)
      #X = Dense(50, activation = "relu")(X)
      Dense(features, activation = "linear",  kernel_initializer = 'normal')
  ])
  #Set up the Optimizers
  sgd = optimizers.SGD(learning_rate)
  adam = optimizers.Adam(lr = learning_rate)
  rmsprop = optimizers.RMSprop(lr = learning_rate)


  #Compile the model
  model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
    
  return model





'''Models for Cross Correlation between Stimuli and EEG'''

def conv_1D_cross_hp(hp):

  
  learning_rate = 0.01
  features = 1
  window = 160
 
  model = Sequential([

    Conv1D(input_shape = (window, features), filters = hp.Int('conv_1_filter', min_value=2, max_value= 6, step= 1),  kernel_size=hp.Choice('conv_1_kernel', values = [3,5])),
    ELU(),
    MaxPooling1D(pool_size= 2),

  
    
    Conv1D(filters =  hp.Int('conv_2_filter', min_value=2, max_value= 6, step= 1 ), kernel_size = hp.Choice('conv_2_kernel', values = [3,5])),
    ELU(),
    MaxPooling1D(pool_size= 2),

    
    Conv1D(filters =  hp.Int('conv_3_filter', min_value=2, max_value= 6, step= 1), kernel_size = hp.Choice('conv_3_kernel', values = [3,5])),
    ELU(),
    MaxPooling1D(pool_size= 2),

    
    Flatten(),
    #out = Activation("linear")(X)
    #X = Dense(50, activation = "relu")(X)
    Dense(features, activation = "linear",  kernel_initializer = 'normal')
  ])
  #Set up the Optimizers
  sgd = optimizers.SGD(learning_rate)
  adam = optimizers.Adam(lr = learning_rate)
  rmsprop = optimizers.RMSprop(lr = learning_rate)


  #Compile the model
  model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
    
  return model


'''Models for Cross Correlation between Stimuli and EEG'''

def conv_1D_cross(dim, source_Y, learning_rate):

    _, window, features = dim

  
    model = Sequential([

    Conv1D(input_shape = (window, features), filters = 2,  kernel_size = 5),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),

  
    
    Conv1D(filters = 4 , kernel_size = 5),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),

    
    Conv1D(filters = 4,  kernel_size = 5),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),

    
    Flatten(),
  
    Dense(features, activation = "linear", kernel_initializer = 'normal')
    ])
    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        
    return model


'''Hybrid Models'''
def conv_lstm( dim, source_Y, learning_rate):
   

    _, window, features = dim

    #Encoder CNN Part 

    model = Sequential([

    Conv1D(input_shape = (window, features), filters = 2,  kernel_size = 5),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),



    Conv1D(filters = 4 , kernel_size = 5),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),


    Conv1D(filters = 4,  kernel_size = 5),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),


    Flatten(),


    #Decoder LSTM part
    RepeatVector(n_outputs),
	LSTM(200, activation='relu', return_sequences=True),
	TimeDistributed(Dense(1)),



    Dense(features, activation = "linear", kernel_initializer = 'normal')
    ])
    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        
    return model



'''Recurent Neural Network Models'''


def vanilla_LSTM(dim,  units, source_Y, cell_type, learning_rate):

    _, window, features = dim
    model = Sequential()
    model.add(Input( (window, features)))
    if cell_type == "LSTM":
        model.add(LSTM(units))
    elif cell_type == "RNN":
        model.add(SimpleRNN(units))
    else:
        model.add(GRU(units))

    model.add(Dense(features))
    #out = Lambda(lambda x: x * 2)(X)
    


    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss= tensorflow.keras.losses.cosine_similarity, optimizer = adam, metrics=['mse'])
        

    return model


'''Tuning of the above model'''


def vanilla_LSTM_hp(hp):

    learning_rate = 0.1
    window = 160
    features = 64

    print("hello")
    
    model = Sequential([

    LSTM(units = hp.Int('LSTM_1_units', min_value=2, max_value= 100, step= 16), input_shape = (window, features), return_sequences = True),
    LSTM(units = hp.Int('LSTM_2_units', min_value=2, max_value= 100, step= 16), return_sequences = True),
    LSTM(units = hp.Int('LSTM_3_units', min_value=2, max_value= 100, step= 16)),
    Dense(features, activation = "linear")
    
    ])

    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        

    return model



def vanilla_LSTM_cross_hp(hp):

    learning_rate = 0.1
    window = 160
    features = 1

    model = Sequential([

    LSTM( hp.Int('LSTM_1_units', min_value=2,max_value= 20, step= 2), input_shape = (None, features)),
    Dense(features, activation = "linear", kernel_initializer = 'normal')
    
    ])

    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        

    return model



def LSTM_autoencoder(dim,  units, source_Y, cell_type, learning_rate):
    _, window, features = dim
   
    
    '''''''''''''''''''''
    Composite LSTM Autoencoder
    
    # define encoder
    visible = Input(shape=(window,features))
    encoder = LSTM(units)(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(window)(encoder)
    decoder1 = LSTM(units, return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(features))(decoder1)

    # define predict decoder
    decoder2 = RepeatVector(window)(encoder)
    decoder2 = LSTM(units, return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(features))(decoder2)
    # tie it together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])

    '''


    '''''''''''''''''''''
    Predictive AutoEncoder
    
    model.add(Input( (window, features)))
    if cell_type == "LSTM":
        model.add(LSTM(units))
    elif cell_type == "RNN":
        model.add(SimpleRNN(units))
    else:
        model.add(GRU(units))

    model.add(RepeatVector(window))
    model.add(LSTM(units,  return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    
    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        

    return model
    
    '''



    '''''''''''''''''''''
    TF AutoEnocoder where the states of the Encoder are propogated to the Decoder Part
    '''''''''''''''''''''
    encoder_inputs = Input(shape=(window, features), name='encoder_inputs')
    decoder_inputs = Input(shape=(window, features), name='decoder_inputs')

    encoder = LSTM(units, return_state=True)
    decoder = LSTM(units, return_sequences=True, return_state=True)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_states = format_encoder_states(cell_type, encoder_states, use_first=False)

    # define training decoder
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(features)
    decoder_outputs = decoder_dense(decoder_outputs)
    # Full encoder-decoder model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='train_model')



    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(units,))
    decoder_state_input_c = Input(shape=(units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states, name='pred_model')

    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        



    # return all models
    return model, encoder_model, decoder_model


def get_decoder_initial_states(units, cell_type):
    """
    Return decoder states as Input layers
    """
    decoder_states_inputs = []

    decoder_state_input_h = Input(shape=(units,))
    input_states = [decoder_state_input_h]
    if cell_type == "LSTM":
        decoder_state_input_c = Input(shape=(units,))
        input_states = [decoder_state_input_h, decoder_state_input_c]
    decoder_states_inputs.extend(input_states)
    if tensorflow.keras.__version__ < '2.2':
        return list(reversed(decoder_states_inputs))
    else:
        return decoder_states_inputs



def build_prediction_model(decoder_inputs, units, cell_type):
    """
    A modified version of the decoder is used for prediction.
    The model takes as inputs:
        - 3D Tensor of shape (batch_size, input_sequence_length, n_features)
        - a 2D Tensor of shape (batch_size, hidden_state) for each layer of the decoder
    and outputs a list containing:
        - the prediction: a 3D tensor of shape (batch_size, 1, 1)
        - a 2D Tensor of shape (batch_size, hidden_state) for each layer of the decoder

    :param decoder_inputs: list
        Predicted target inputs (np.array (batch_size, 1, n_features))
    :return:
    """
    decoder_inputs = Input(shape=decoder_inputs)
    decoder_states_inputs = get_decoder_initial_states(units, cell_type)

    decoder = LSTM(units, return_sequences=True, return_state=True)
    decoder_dense = Dense(1)

    decoder_outputs = decoder(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = decoder_outputs[1:]
    decoder_outputs = decoder_outputs[0]
    decoder_outputs = decoder_dense(decoder_outputs)

    # Decoder model to be used during inference
    decoder_pred = Model([decoder_inputs] + decoder_states_inputs,
                                [decoder_outputs] + decoder_states,
                                name='pred_model')
    return decoder_pred


def format_encoder_states(cell_type, encoder_states, use_first=True):
    """
    Format the encoder states in such a way that only the last state from the first layer of the encoder
    is used to init the first layer of the decoder.
    If the cell type used is LSTM then both c and h are kept.
    :param encoder_states: tensorflow.keras.tensor
        (last) hidden state of the decoder
    :param use_first: bool
        if True use only the last hidden state from first layer of the encoder, while the other are init to zero.
        if False use last hidden state for all layers
    :return:
        masked encoder states
    """
    if use_first:
        # tensorflow.keras version 2.1.4 has encoder states reversed w.r.t later versions
        if tensorflow.keras.__version__ < '2.2':
            if cell_type == 'LSTM':
                encoder_states = [Lambda(lambda x: tensorflow.keras.zeros_like(x))(s) for s in encoder_states[:-2]] + [
                    encoder_states[-2]]
            else:
                encoder_states = [Lambda(lambda x: tensorflow.keras.zeros_like(x))(s) for s in encoder_states[:-1]] + [
                    encoder_states[-1]]
        else:
            if cell_type == 'LSTM':
                print("Fuck OFF")
                encoder_states = encoder_states[:2] + [Lambda(lambda x: tensorflow.keras.zeros_like(x))(s) for s in
                                                            encoder_states[2:]]
            else:
                encoder_states = encoder_states[:1] + [Lambda(lambda x: tensorflow.keras.zeros_like(x))(s) for s in
                                                            encoder_states[1:]]
    return encoder_states




def get_model():

  MODELS = {"LR":linear_regression,
            "LSTM": vanilla_LSTM, 
            "CNN": conv_1D, 
            "CNN_hp":conv_1D_hp,
            "LSTM_hp":vanilla_LSTM_hp,
            "LSTM_cross_hp": vanilla_LSTM_cross_hp,
            "CNN_cross_hp":conv_1D_cross_hp,
            "CNN_cross":conv_1D_cross,
            "LSTM_autoencoder":LSTM_autoencoder,
            "conv_lstm":conv_lstm
            }
  return MODELS