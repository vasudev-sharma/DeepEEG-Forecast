import os
from keras import optimizers
from input import data
from models import get_model
from metrics import compute_correlation


if __name__ == "__main__":
    train, valid, test = data()

    train_X, train_Y = train
    valid_X, valid_Y = valid
    test_X, test_Y = test

    #Parameters of model
    training_epochs = 100
    batch_size = train_X.shape[0]
    layers = 1
    units = 3
    learning_rate = 0.1
    

    #Set up the Optimizers
    sgd = optimizers.SGD(lr = learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)

    model = get_model()["LR"]
    model = model(train_X.shape)

    #Compile the model4
    model.compile(loss = 'mse', optimizer = sgd, metrics= ['mse'])

    print(model.summary())

    # Fit the model with the Data
    history = model.fit(
        train_X, 
        train_Y, 
        batch_size = batch_size,
        epochs = training_epochs, 
        validation_data = (valid_X, valid_Y), 
        verbose = 1,
        )

    predictions = model.predict(test_X, verbose = 1)

    print("The value of correlation is {}".format(compute_correlation(predictions, test_Y)))