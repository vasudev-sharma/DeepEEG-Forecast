
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers
from tensorflow.keras import optimizers



'''Linear Regression Models'''

def linear_regression(dim, learning_rate):

    _, features = dim 
    out_features = features / 160
    inp = Input((features,))

    X = inp
    out = Dense(out_features, activation = "linear", kernel_initializer = "normal" )(X)
    model = Model(inputs = inp , output = out)

     #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])

    return model 





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




#RNN
def vanilla_RNN(dim,  units, source_Y, learning_rate):

    _, window, features = dim
    inp = Input([window, features])
    X = inp
 
    X = SimpleRNN(units)(X)
    out = Dense(source_Y, activation = "linear", kernel_initializer = 'normal')(X)
    
    model = Model(inputs = inp, outputs = out)

     #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])

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
    LeakyReLU(),
    MaxPooling1D(pool_size= 2),

  
    
    Conv1D(filters =  hp.Int('conv_2_filter', min_value=2, max_value= 6, step= 1 ), kernel_size = hp.Choice('conv_2_kernel', values = [3,5])),
    LeakyReLU(),
    MaxPooling1D(pool_size= 2),

    
    Conv1D(filters =  hp.Int('conv_3_filter', min_value=2, max_value= 6, step= 1), kernel_size = hp.Choice('conv_3_kernel', values = [3,5])),
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





'''Recurent Neural Network Models'''


def vanilla_LSTM(dim,  units, source_Y, cell_type, learning_rate):

    _, window, features = dim
    inp = Input([window, features])
    X = inp
    if cell_type == "LSTM":
        X = LSTM(units)(X)
    elif cell_type == "RNN":
        X = SimpleRNN(units)(X)
    else:
        X = GRU(units)(X) 

    out = Dense(source_Y, activation = "linear", kernel_initializer = 'normal')(X)
    #out = Lambda(lambda x: x * 2)(X)
    
    model = Model(inputs = inp, outputs = out)

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

  
    
    model = Sequential([

    LSTM(hp.Int('LSTM_1_units', min_value=2, max_value= 150, step= 16)),
    Dense(features, activation = "linear", kernel_initializer = 'normal')
    
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

    LSTM(hp.Int('LSTM_1_units', min_value=2, max_value= 20, step= 2)),
    Dense(features, activation = "linear", kernel_initializer = 'normal')
    
    ])

    #Set up the Optimizers
    sgd = optimizers.SGD(learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model
    model.compile(loss = 'mse', optimizer = sgd, metrics=['mse'])
        

    return model







def get_model():

  MODELS = {"LR":linear_regression,
            "LSTM": vanilla_LSTM, 
            "CNN": conv_1D, 
            "CNN_hp":conv_1D_hp,
            "LSTM_hp":vanilla_LSTM_hp,
            "LSTM_cross_hp": vanilla_LSTM_cross_hp,
            "CNN_cross_hp":conv_1D_cross_hp,

            }
  return MODELS