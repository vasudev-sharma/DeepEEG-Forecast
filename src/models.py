
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import Ini



def linear_regression(X_shape):
    X = Input((X_shape,))
    rng = initializers.random_normal(0, 1)
    out = Dense(1, activation = "linear", kernel_initializer = "normal" )(X)
    model = Model(inputs = X , output = out)
    return model 

def conv_1D(layers, features):
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