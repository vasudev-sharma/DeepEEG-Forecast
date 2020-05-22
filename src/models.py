
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
import tensorflow.keras
import tensorflow
from metrics import cosine_loss, mean_squared_loss
from tensorflow.keras.regularizers import L1L2
from collections import UserDict, deque


'''Linear Regression Models'''

def linear_regression(dim, learning_rate, loss, optimizer):

    _, features = dim 
    out_features = 160

  

    model = Sequential([
    Dense( int(out_features), input_shape = (features,) ,activation = "linear" )])

     #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)

    
    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSLE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam

    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])

    return model 



'''CNN Models'''

#COnvolutional Neural Network
def conv_1D(dim, source_Y, learning_rate, loss, optimizer):

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

    
    model = Model(inputs = inp, outputs = out)
     #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)
    adagrad = optimizers.Adagrad(lr =learning_rate)

    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam


    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])

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
  model.compile(loss = tensorflow.keras.losses.MeanSquaredError(), optimizer = adam, metrics=['mse'])
    
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
  model.compile(loss = tensorflow.keras.losses.MeanSquaredError(), optimizer = adam, metrics=['mse'])
    
  return model


'''Models for Cross Correlation between Stimuli and EEG'''

def conv_1D_cross(dim, source_Y, learning_rate, loss, optimizer):

    _, window, features = dim

    print(features)
    model = Sequential([

    Conv1D(input_shape = (window, features) ,filters = 2,  kernel_size = 5 ),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),

  
    
    Conv1D(filters = 4  , kernel_size = 3),
 
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),

    
    Conv1D(filters = 4 ,kernel_size = 3),
    ELU(),
    SpatialDropout1D(0.1),
    MaxPooling1D(pool_size= 2),


    Flatten(),
   
    Dense(160, activation = "linear", kernel_initializer = 'normal')
    
    ])
    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)
    adagrad = optimizers.Adagrad(lr =learning_rate)

    print("Value of Optimzer is", optimizer)

    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    if optimizer == "Adam":
        print("Adam optimizer is used")
        optimizer = adam

    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])

    return model




'''Recurent Neural Network Models'''


def vanilla_LSTM(dim,  units, source_Y, cell_type, learning_rate, loss, optimizer):

    _, window, features = dim
    model = Sequential()
    model.add(Input( (window, features)))
    print(features)
    if cell_type == "LSTM":
        model.add(LSTM(units))
        #model.add(LSTM(units, return_sequences= True))
    elif cell_type == "RNN":
        model.add(SimpleRNN(units))
    else:
        model.add(GRU(units))

    model.add(Dense(1))
    


    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)

    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam


    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])


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
    model.compile(loss = tensorflow.keras.losses.MeanSquaredError(), optimizer = adam, metrics=['mse'])
        

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
    model.compile(loss = tensorflow.keras.losses.MeanSquaredError(), optimizer = adam, metrics=['mse'])
        

    return model


########################
#Hybrid Models
########################

def conv_lstm( dim, source_Y, learning_rate, loss, optimizer):
   

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


    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam


    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])

    return model




def combined_model(dim,  units, source_Y, cell_type, learning_rate, loss, optimizer):
   

    _, window, features = dim

  
    #CNN Model

    input_1 = Input((window, 1))
    X = input_1


    X = Conv1D(input_shape = (window, features), filters = 2,  kernel_size = 5)(X)
    X = ELU()(X)
    X = SpatialDropout1D(0.1)(X)
    X = MaxPooling1D(pool_size= 2)(X)



    X = Conv1D(filters = 4 , kernel_size = 5)(X)
    X = ELU()(X)
    X = SpatialDropout1D(0.1)(X)
    X = MaxPooling1D(pool_size= 2)(X)


    X = Conv1D(filters = 4,  kernel_size = 5)(X)
    X = ELU()(X)
    X = SpatialDropout1D(0.1)(X)
    X = MaxPooling1D(pool_size= 2)(X)


    X = Flatten()(X)
  


    #LSTM Model
    input_2 = Input((window, 1))
    Y = input_2
    print(features)
    if cell_type == "LSTM":
        Y = LSTM(units)(Y)
    elif cell_type == "RNN":
        Y = SimpleRNN(units)(Y)
    else:
        Y = GRU(units)(Y)

     # Concatenate
    concat = tensorflow.keras.layers.Concatenate()([X, Y])

    
    output = Dense(1)(concat)
    #out = Lambda(lambda x: x * 2)(X)
    


    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)
    
    model = Model([input_1, input_2], output)


    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam

    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])
        
    return model





def LSTM_autoencoder(dim,  units, source_Y, cell_type, learning_rate, teacher_force, loss, optimizer):
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

    ################################################
    #LSTM Seq2Seq without and with Teacher Forcing
    ################################################
   
    encoder_inputs = Input(shape=(window, features), name='encoder_inputs')
    teacher_force = teacher_force
    z_score_outputs = False
    if teacher_force:
      print("Model is using teacher forcing")
      decoder_inputs = Input(shape=(1, features), name='decoder_inputs')
    else:
      print("Model is not using teacher forcing")
      decoder_inputs = Input(shape=(None, features), name='decoder_inputs')

    if teacher_force:
        encoder = LSTM(units, return_state=True)
    else:
        encoder = LSTM(units, return_state=True)

    decoder = LSTM(units, return_sequences=True, return_state=True)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    encoder_model = Model(encoder_inputs, encoder_states)

    #encoder_states = format_encoder_states(cell_type, encoder_states, use_first=False)

    # define training decoder
    if not teacher_force:
      decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
      decoder_dense = Dense(1)
      decoder_outputs = decoder_dense(decoder_outputs)
    else:
      decoder_outputs = build_static_loop(encoder_states, decoder_inputs, decoder)
    
    if z_score_outputs:
      decoder_outputs = tensorflow.math.divide(tensorflow.math.subtract(decoder_outputs, tensorflow.keras.backend.mean(decoder_outputs,axis=1,keepdims=True)),tensorflow.keras.backend.std(decoder_outputs,axis=1,keepdims=True))
    
    if teacher_force:
        # Full encoder-decoder model
        model = Model(encoder_inputs, decoder_outputs, name='train_model')
    else:
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='train_model')


    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)
    

    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam

    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])

    # return all models
    return model, encoder_model



###################################################
#Help Functions for Encoder Decoder Architecture
###################################################


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
                encoder_states = [Lambda(lambda x: tensorflow.keras.backend.zeros_like(x))(s) for s in encoder_states[:-2]] + [
                    encoder_states[-2]]
            else:
                encoder_states = [Lambda(lambda x: tensorflow.keras.backend.zeros_like(x))(s) for s in encoder_states[:-1]] + [
                    encoder_states[-1]]
        else:
            if cell_type == 'LSTM':
                print("Fuck OFF")
                encoder_states = encoder_states[:2] + [Lambda(lambda x: tensorflow.keras.backend.zeros_like(x))(s) for s in
                                                            encoder_states[2:]]
            else:
                encoder_states = encoder_states[:1] + [Lambda(lambda x: tensorflow.keras.backend.zeros_like(x))(s) for s in
                                                            encoder_states[1:]]
    return encoder_states



def build_static_loop(init_states, decoder_inputs, decoder):
    """
    :param init_states: list
        list og length = number of layers of encoder/decoder.
        Each element is a 2D tensor of shape (batch_size, units)
    :param decoder_inputs:
        3D tensor of shape (batch_size, 1, 1)
    :param decoder_inputs_exog:
        3D tensor of shape (batch_size, output_sequence_length, n_features - 1)
    :return:
        3D tensor of shape (batch_size, output_sequence_length, 1)
    """
    horizon = 160
    decoder_dense = Dense(1)
    inputs = decoder_inputs
    all_outputs = []
    for i in range(horizon):
        decoder_outputs = decoder(inputs, initial_state=init_states)
        init_states = decoder_outputs[1:] # state update
        decoder_outputs = decoder_outputs[0]
        decoder_outputs = decoder_dense(decoder_outputs)  # (batch, 1, 1)
        all_outputs.append(decoder_outputs)
        inputs = decoder_outputs # input update

    decoder_outputs = Lambda(lambda x: tensorflow.keras.layers.concatenate(x, axis=1))(all_outputs)
    return decoder_outputs




def ES_RNN(dim, units, source_Y, cell_type, learning_rate, loss, optimizer, batch_size):

    _, window, features = dim
    horizon = 160
    m = 180 #Seasonality length
    model_input = Input(shape=(window, features))
    [normalized_input, denormalization_coeff] = ES(horizon, m, batch_size, window)(model_input)
    gru_out = GRU(units)(normalized_input)
    model_output_normalized = Dense(horizon)(gru_out)
    model_output = Denormalization()([model_output_normalized, denormalization_coeff])
    model = Model(inputs=model_input, outputs=model_output)

    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)

    if loss == "MSE":
        print("MSE Loss is used")
        loss = tensorflow.keras.losses.MSE
    else:
        print("Cosine loss is used")
        loss = cosine_loss
    
    if optimizer == "SGD":
        print("SGD optimizer is used")
        optimizer = sgd
    else:
        print("Adam optimizer is used")
        optimizer = adam


    #Compile the model
    model.compile(loss = loss, optimizer = optimizer, metrics=['mse'])


    return model





# Exponential Smoothing + Normalization
class ES(Layer):

    def __init__(self, horizon, m, batch_size, time_steps, **kwargs):
        self.horizon = horizon
        self.m = m
        self.batch_size = batch_size
        self.time_steps = time_steps
        
        super(ES, self).__init__(**kwargs)

    # initialization of the learned parameters of exponential smoothing
    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(1,),
                                     initializer='uniform', trainable=True) 
        self.gamma = self.add_weight(name='gamma', shape=(1,),
                                     initializer='uniform', trainable=True)
        self.init_seasonality = self.add_weight(name='init_seasonality', shape=(self.m,),
                                                initializer=initializers.Constant(value=0.8), trainable=True)
        self.init_seasonality_list = [tensorflow.slice(self.init_seasonality,(i,),(1,)) for i in range(self.m)]
        self.seasonality_queue = deque(self.init_seasonality_list, self.m)
        self.level = self.add_weight(name='init_level', shape=(1,),
                                     initializer=initializers.Constant(value=0.8), 
                                     trainable=True)
        super(ES, self).build(input_shape)  


    def call(self, x):

        # extract time-series from feature vector
        n_examples = tensorflow.keras.backend.int_shape(x)[0]
        if n_examples is None:
            n_examples = self.batch_size
        x1 = tensorflow.slice(x,(0,0,0),(1,self.time_steps,1))
        x1 = tensorflow.keras.backend.reshape(x1,(self.time_steps,))
        x2 = tensorflow.slice(x,(1,self.time_steps-1,0),(n_examples-1,1,1))
        x2 = tensorflow.keras.backend.reshape(x2,(n_examples-1,))
        ts = tensorflow.keras.backend.concatenate([x1,x2])
        
        x_norm = []  # normalized values of time-series
        ls = []      # coeffients for denormalization of forecasts
        
        l_t_minus_1 = self.level
        
        for i in range(n_examples+self.time_steps-1):
        
            # compute l_t
            y_t = ts[i]
            s_t = self.seasonality_queue.popleft()
            l_t = self.alpha * y_t / s_t + (1 - self.alpha) * l_t_minus_1
            
            # compute s_{t+m}
            s_t_plus_m = self.gamma * y_t / l_t + (1 - self.gamma) * s_t
            
            self.seasonality_queue.append(s_t_plus_m)
            
            # normalize y_t
            x_norm.append(y_t / (s_t * l_t))

            l_t_minus_1 = l_t

            if i >= self.time_steps-1:
                l = [l_t]*self.horizon
                l = tensorflow.keras.backend.concatenate(l)
                s = [self.seasonality_queue[i] for i in range(self.horizon)] # we assume here that horizon < m
                s = tensorflow.keras.backend.concatenate(s)
                ls_t = tensorflow.keras.backend.concatenate([tensorflow.keras.backend.expand_dims(l), tensorflow.keras.backend.expand_dims(s)])
                ls.append(tensorflow.keras.backend.expand_dims(ls_t,axis=0))  
       
        self.level = l_t
        x_norm = tensorflow.keras.backend.concatenate(x_norm)

        # create x_out
        x_out = []
        for i in range(n_examples):
            norm_features = tensorflow.slice(x_norm,(i,),(self.time_steps,))
            norm_features = tensorflow.keras.backend.expand_dims(norm_features,axis=0)
            x_out.append(norm_features)

        x_out = tensorflow.keras.backend.concatenate(x_out, axis=0)
        x_out = tensorflow.keras.backend.expand_dims(x_out)

        # create tensor of denormalization coefficients 
        denorm_coeff = tensorflow.keras.backend.concatenate(ls, axis=0)
        return [x_out, denorm_coeff]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], input_shape[2]), (input_shape[0], self.horizon, 2)]
    
class Denormalization(Layer):
    
    def __init__(self, **kwargs):
        super(Denormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Denormalization, self).build(input_shape)  

    def call(self, x):
        return x[0] * x[1][:,:,0] * x[1][:,:,1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]










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
            "conv_lstm":conv_lstm,
            "combined_model":combined_model,
            "ES_RNN":ES_RNN
            }
  return MODELS