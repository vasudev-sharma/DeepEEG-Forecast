import os
import json
from keras import optimizers
from input import data
from models import get_model
from metrics import compute_correlation
from keras.callbacks import ReduceLROnPlateau

pred = os.environ["pred"]


if __name__ == "__main__":
    
    

    print("The predicted value is ", pred)
    train, valid, test = data(int(pred), "3", "2")

    train_X, train_Y = train
    valid_X, valid_Y = valid
    test_X, test_Y = test

    #Read the parameters of the model
    with open("parameters.json", "r") as param_file:
        parameters = json.load(param_file)

    #Parameters of model
    training_epochs = parameters["training_epochs"]
    batch_size = parameters["batch_size"]
    layers = parameters["layers"]
    units = parameters["units"]
    learning_rate = parameters["learning_rate"]
    
    '''
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=0.00001)
    '''

    #Set up the Optimizers
    sgd = optimizers.SGD(lr = learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)

    model = get_model()["CNN"]
    model = model(train_X.shape, layers, train_Y.shape[-1])


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
    #Predict the Y values for the given test set
    predictions = model.predict(test_X, verbose = 1)

    #Compute Correlation coefficient 
    corr = compute_correlation(predictions, test_Y)
    print("The value of correlation is for electrode {} is {}". format(pred, corr))


    #Dump the values in json file
    data= {"Electrode_"+pred:corr}
    with open("corr_dat.json", "a") as write_file:
        json.dump(data, write_file)
