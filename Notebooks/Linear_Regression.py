#!/usr/bin/env python
# coding: utf-8

# In[4]:



from keras.layers import *
from keras.models import *
from keras import initializers

# In[5]:


def linear_regression(X_shape):
    X = Input((X_shape,))
    rng = initializers.random_uniform(0, 1)
    out = Dense(1, activation = "linear", kernel_initializer = rng )(X)
    model = Model(inputs = X , output = out)
    return model 

    



