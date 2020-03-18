import os
from tensorflow.keras import optimizers


DATA = os.environ.get("DATA")
model = os.environ.get("MODEL")


#


if __name__ == "__main__":
    train, valid, test = DATA

    train_X, train_Y = train
    valid_X, valid_Y = valid
    test_X, test_Y = test



    #Set up the Optimizers
    sgd = optimizers.SGD(lr = learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)


    #Compile the model4
    model.compile(loss = 'mse', optimizer = sgd, metrics= ['mse'])



    # Fit the model with the Data
    history = model.fit(
        train_X, 
        train_Y, 
        batch_size = batch_size,
        epochs = training_epochs, 
        validation_data = (valid_X, valid_Y), 
        verbose = 1,
        )

