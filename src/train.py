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

    #Parameters of model
    training_epochs = 200
    batch_size = 512
    layers = 1
    units = 3
    learning_rate = 0.005
    

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=0.00001)
    

    #Set up the Optimizers
    sgd = optimizers.SGD(lr = learning_rate)
    adam = optimizers.Adam(lr = learning_rate)
    rmsprop = optimizers.RMSprop(lr = learning_rate)

    model = get_model()["CNN"]
    model = model(train_X.shape, layers, train_Y.shape[-1])

    #Compile the model4
    model.compile(loss = 'mse', optimizer = sgd, metrics= ['mse'], )

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
    predictions = model.predict(test_X, verbose = 1,  callbacks=[reduce_lr])

    #Compute Correlation coefficient 
    corr = compute_correlation(predictions, test_Y)
    print("The value of correlation is for electrode {} is {}", format(pred, corr))


    #Dump the values in json file
    data= {"Electrode_"+pred:corr}
    with open("corr_dat.json", "a") as write_file:
        json.dump(data, write_file)
