
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers
from tensorflow.keras import optimizers



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

    model.add(Dense(features, activation = "linear"))
    #out = Lambda(lambda x: x * 2)(X)
    


    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        

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
    model = Sequential()
    

    '''
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

    #model.add(Dense(source_Y, activation = "linear"))
    #out = Lambda(lambda x: x * 2)(X)
    


    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        

    return model
    '''

    encoder_inputs = Input(shape=encoder_inputs, name='encoder_inputs')
    decoder_inputs = Input(shape=decoder_inputs, name='decoder_inputs')



def predict(encoder_model, decoder_model, encoder_inputs, pred_steps):
    """
    Multi step Inference (1 at a time)
    :param encoder_inputs: numpy.array
        Encoder input: shape(n_samples, input_sequnece_length, n_features)
    :param pred_steps: int
        number of steps to be predicted in the future
 
    :return: numpy.array
        shape(n_samples, output_sequence_length, 1)
    """
    # predictions, shape (batch_size, pred_steps, 1)
    predictions = np.zeros((encoder_inputs.shape[0], pred_steps, 1))

    # produce embeddings with encoder
    states_value = encoder_model.predict(encoder_inputs)  # [h,c](lstm) or [h](gru) each of dim (batch_size, n_hidden)

    # populate the decoder input with the last encoder input
    decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))  # decoder input for a single timestep
    decoder_input[:, 0, 0] = encoder_inputs[:, -1, 0]

    for i in range(pred_steps):
        

        if isinstance(states_value, list):
            outputs = decoder_pred.predict([decoder_input] + states_value)
        else:
            outputs = decoder_pred.predict([decoder_input, states_value])

        # prediction at timestep i
        output = outputs[0]  # output (batch_size, 1, 1)
        predictions[:, i, 0] = output[:, 0, 0]

        # Update the decoder input with the predicted value (of length 1).
        decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))
        decoder_input[:, 0, 0] = output[:, 0, 0]

        # Update states
        states_value = outputs[1:] # h, c (both [batch_size, n_hidden]) or just h

    return predictions




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