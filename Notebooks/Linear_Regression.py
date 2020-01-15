#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras import backend as K
from keras.layers import *
from keras.models import *


# In[5]:


def linear_regression(X_shape):
    X = Input((X_shape,))
    out = Dense(1, activation = "linear")(X)
    model = Model(inputs = X , output = out)
    return model 

    


# In[ ]:




