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
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import random 
import string

pred = os.environ["pred"]
stimulus = os.environ["stimulus"]
relation = os.environ["relation"]
model_name = os.environ["model_name"]


if __name__ == "__main__":
    
    # Set the Training parameter to False to True whether you want training 
    training =  True
    model_name = model_name

    #No of predictions steps ahead
    horizon = 160

    multivariate = False
    split = True
    if horizon > 1:
        multivariate = True
    if model_name == "LR":
        split = False 
    
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

            flag_tuning = False   # Set the hyper param tuning if you want to tune number of layer, model etc
            
            model = get_model()[model_name]
            if model_name == "LSTM":
                model = model(train_X.shape, units, train_Y.shape[-1], learning_rate)
            elif model_name == "CNN":
                model = model(train_X.shape, units, train_Y.shape[-1], learning_rate)
            elif model_name =="LR":
                model = model(train_X.shape, learning_rate)
            else:
                flag_tuning = True
                model = model 
            
            if flag_tuning == True:
                training_epochs = 5
                random_string = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])

                tuner_search=RandomSearch(model,
                                        objective='val_loss',
                                        max_trials=20,project_name= "experiment/"+ random_string,
                                        executions_per_trial=1,
                )

                tuner_search.search(train_X, 
                    train_Y, 
                    batch_size = batch_size,
                    epochs = training_epochs, 

                    validation_data = (valid_X, valid_Y))

                tuner_search.results_summary()
                sys.exit()

            else:

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

            model = load_model('../models/{}/model_{}_all_channel.h5'.format(model_name, model_name))
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

        

        if pred!=-1:
            #Dump the values in json file
            data= {"Electrode_"+pred:corr}
            with open("corr_dat.json", "a") as write_file:
                json.dump(data, write_file)
            

