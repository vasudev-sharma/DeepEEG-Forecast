
from keras.layers import *
from keras.models import Model
from keras import initializers



#Linear Regression Model
def linear_regression(dim):

    _, features = dim 
    out_features = features / 160
    inp = Input((features,))

    X = inp
    out = Dense(out_features, activation = "linear", kernel_initializer = "normal" )(X)
    model = Model(inputs = inp , output = out)
    return model 


#COnvolutional Neural Network
def conv_1D(dim, layers, source_Y):

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
    return model


#RNN
def vanilla_RNN(dim, layers, units, source_Y):

    _, window, features = dim
    inp = Input([window, features])
    X = inp
    for _ in range(layers - 1):
      X = SimpleRNN(units, return_sequences = True)(X)
    X = SimpleRNN(units)(X)
    out = Dense(source_Y, activation = "linear", kernel_initializer = 'normal')(X)
    
    model = Model(inputs = inp, outputs = out)
    return model



#RNN
def vanilla_LSTM(dim, layers, units, source_Y):

    _, window, features = dim
    inp = Input([window, features])
    X = inp
    for _ in range(layers - 1):
      X = LSTM(units, return_sequences = True)(X)
    X = LSTM(units)(X)
    out = Dense(source_Y, activation = "linear", kernel_initializer = 'normal')(X)
    #out = Lambda(lambda x: x * 2)(X)
    
    model = Model(inputs = inp, outputs = out)
    return model


def get_model():

  MODELS = {"LR":linear_regression,
            "LSTM": vanilla_LSTM, 
            "CNN": conv_1D, 
            "RNN":vanilla_RNN}
  return MODELS