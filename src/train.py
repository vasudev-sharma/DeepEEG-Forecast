import os
import json
from tensorflow.keras import optimizers
from input import data
from tensorflow.keras.models import load_model
from predict import predict_single_timestep, predict_multi_timestep
from models import get_model
from metrics import compute_correlation, list_correlation
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import plot_multistep_prediction
from numpy import savez_compressed

pred = os.environ["pred"]
stimulus = os.environ["stimulus"]
relation = os.environ["relation"]


if __name__ == "__main__":
    
    # Set the Training parameter to False to True whether you want training 
    training = True
    model_name = "CNN"

    #No of predictions steps ahead
    horizon = 1

    multivariate = False
    split = True
    if horizon > 1:
        multivariate = True
    if model_name == "LR":
        split = False 
    

    print("The predicted value is ", pred)
    train, valid, test = data(int(pred), relation= relation, stimulus= stimulus, horizon = horizon,  split = split , multivariate = multivariate)

    train_X, train_Y = train
    valid_X, valid_Y = valid
    test_X, test_Y = test

    
    if training: 

        #Read the parameters of the model
        with open("../config/{}/parameters.json".format(model_name), "r") as param_file:
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

        
        model = get_model()[model_name]
        model = model(train_X.shape, train_Y.shape[-1])


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

        model.save('../models/{}/{}.h5'.format(model_name, model_name))

    else: 

        model = load_model('../models/{}/model_LSTM_all_channel.h5'.format(model_name))
        print(model.summary())

    
    #Predict the Y values for the given test set
    predictions = predict_multi_timestep(model, test_X, horizon = horizon, model_name = model_name)

    #plot_multistep_prediction(test_Y, predictions )

    #Actual and Predicted values for Single electrode mutistep 
    true = test_Y[:, :, 63]
    pred = predictions[:, :, 63]

    corr = list_correlation(true, pred)
    print("The value of correlation is for electrode 63 is {}". format(corr))


    '''Save the predicted and True values in the numpy array'''
    savez_compressed('/root/EEG/models/{}/True.npz'.format(model_name), test_Y)
    savez_compressed('/root/EEG/models/{}/predicted.npz'.format(model_name), predictions)



    '''

    #Compute Correlation coefficient 
    corr = compute_correlation(predictions, test_Y)
    print("The value of correlation is for electrode {} is {}". format(pred, corr))

    '''

    



    '''
    Not working
    #Dump the values in json file
    data= {"Electrode_"+pred:corr}
    with open("corr_dat.json", "a") as write_file:
        json.dump(data, write_file)
    '''