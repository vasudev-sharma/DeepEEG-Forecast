
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import initializers


#Linear Regression Model
def linear_regression(X_shape):
    X = Input((X_shape,))
    rng = initializers.random_normal(0, 1)
    out = Dense(1, activation = "linear", kernel_initializer = "normal" )(X)
    model = Model(inputs = X , output = out)
    return model 


#COnvolutional Neural Network
def conv_1D(window, layers, features):
    inp = Input([window, features])
    X = inp
    
    X = Conv1D(filters = 16, kernel_size = 5)(X)
    X = LeakyReLU()(X)
    X = MaxPooling1D(pool_size= 2)(X)

  
  
    X = Conv1D(filters = 40, kernel_size = 5)(X)
    X = LeakyReLU()(X)
    X = MaxPooling1D(pool_size= 2)(X)


    X = Conv1D(filters = 40, kernel_size = 5)(X)
    X = LeakyReLU()(X)
    X = MaxPooling1D(pool_size= 2)(X)
    
    X = Flatten()(X)
    #out = Activation("linear")(X)
    #X = Dense(50, activation = "relu")(X)
    out = Dense(len(source_Y), activation = "linear",  kernel_initializer = 'normal')(X)
    
    model = Model(inputs = inp, outputs = out)
    return model


#RNN
def vanilla_RNN(layers, units, features):
    inp = Input([stim, features])
    X = inp
    for i in range(layers - 1):
      X = SimpleRNN(units, return_sequences = True)(X)
    X = SimpleRNN(units)(X)
    out = Dense(len(source_Y), activation = "linear", kernel_initializer = 'normal')(X)
    #out = Lambda(lambda x: x * 2)(X)
    
    model = Model(inputs = inp, outputs = out)
    return model



#RNN
def vanilla_LSTM(layers, units, features):
    inp = Input([stim, features])
    X = inp
    for i in range(layers - 1):
      X = LSTM(units, return_sequences = True)(X)
    X = LSTM(units)(X)
    out = Dense(len(source_Y), activation = "linear", kernel_initializer = 'normal')(X)
    #out = Lambda(lambda x: x * 2)(X)
    
    model = Model(inputs = inp, outputs = out)
    return model

