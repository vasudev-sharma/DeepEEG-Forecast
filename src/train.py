import os
import json
from tensorflow.keras import optimizers
from input import data, teacher_forcing
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from predict import predict_single_timestep, predict_multi_timestep, predict_autoencoder
from models import get_model, build_prediction_model
from metrics import compute_correlation, list_correlation
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import plot_multistep_prediction, plot_loss_curve
from numpy import savez_compressed
import numpy as np
from tqdm import tqdm
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import sys
from utils import plot_weights

import random 
import string

pred = os.environ["pred"]
stimulus = os.environ["stimulus"]
input_task = os.environ["input_task"]
model_name = os.environ["model_name"]
horizon = os.environ["horizon"]
training = os.environ["training"]
MIMO_output = os.environ["MIMO_output"]
experiment_no=os.environ["experiment_no"]



if __name__ == "__main__":
    
    # Set the Training parameter to False to True whether you want training 

    training =  training
    model_name = model_name
    

    flag_tuning = True if model_name.endswith("hp") else False


    #No of predictions steps ahead
    horizon = int(horizon)

    #Window of the data
    window = 160

    multivariate = False
    split = True


    if horizon > 1:
        multivariate = True
    if model_name == "LR":
        split = False 
        

    

    print("The predicted value is ", pred)
    train, valid, test = data(int(pred), input_task= input_task, stimulus= stimulus, horizon = horizon,  split = split , multivariate = multivariate)

    train_X, train_Y = train
    valid_X, valid_Y = valid
    test_X, test_Y = test        

    if model_name == "LSTM_autoencoder":
      encoder_input_train, decoder_input_train, decoder_target_train = teacher_forcing(train_X, train_Y)
      encoder_input_valid, decoder_input_valid,  decoder_target_valid = teacher_forcing(valid_X, valid_Y)
      encoder_input_test, decoder_input_test,  decoder_target_test = teacher_forcing(test_X, test_Y)
      input_train = [encoder_input_train, decoder_input_train]
      input_valid = [encoder_input_valid, decoder_input_valid]
      output_train = decoder_target_train
      output_valid = decoder_target_valid

      print("Shape of encoder_input_train, decoder_input_train, decoder_target_train is  ",  encoder_input_train.shape, decoder_input_train.shape, decoder_target_train.shape)
      print("Shape of encoder_input_valid, decoder_input_valid,  decoder_target_valid is ", encoder_input_valid.shape, decoder_input_valid.shape,  decoder_target_valid.shape)  
     


    
    if training: 

            #Read the parameters of the model
            with open("../config/{}/parameters.json".format(model_name), "r") as param_file:
                parameters = json.load(param_file)

            #Parameters of model
            training_epochs = parameters["training_epochs"]
            if model_name == "LR" and input_task=="1":
                batch_size = train_X.shape[0]
            else:     
                batch_size = parameters["batch_size"]
            
            units = parameters["units"]
            learning_rate = parameters["learning_rate"]
            if(model_name == "LSTM" or model_name=="LSTM_hp" or model_name =="LSTM_autoencoder" or model_name=="conv_LSTM" or model_name == "combined_model"):
                cell_type = parameters["cell_type"]
            


            '''
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                    patience=10, min_lr=0.00001)
            '''
            ####################################
            # Callbacks    
            ###################################
            callback_early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
            callback_checkpoint = ModelCheckpoint("../models/{}/{}_best_model.h5".format(model_name, model_name), monitor='val_loss', save_best_only=True, verbose = 1)
        
            


            ############################
            #Compile the model 
            ############################
            model = get_model()[model_name]
            if model_name == "LSTM" or model_name=="conv_LSTM" or model_name == "combined_model":
                model = model(train_X.shape, units, train_Y.shape[-1], cell_type, learning_rate)
            elif model_name =="LSTM_autoencoder":
                model, encoder_model, decoder_model = model(train_X.shape, units, train_Y.shape[-1], cell_type, learning_rate)
            elif model_name == "CNN" or model_name =="CNN_cross":
                model = model(train_X.shape, train_Y.shape[-1], learning_rate)
            elif model_name =="LR":
                model = model(train_X.shape, learning_rate)
            else:
                model = model 
            



            if flag_tuning == True:
                random_string = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])

                tuner_search=RandomSearch(model,
                                        objective='val_loss',
                                        max_trials=40,project_name= "experiment/"+ random_string,
                                        executions_per_trial=1,
                )

                tuner_search.search(train_X, 
                    train_Y, 
                    batch_size = batch_size,
                    epochs = training_epochs, 

                    validation_data = (valid_X, valid_Y))

                tuner_search.results_summary()

                #Use the best model
                model=tuner_search.get_best_models(num_models=1)[0]
                
                training_epochs = 30


            print(model.summary())
            # Fit the model with the Data
            plot_model(model, "{}_model.png".format(model_name), True, True)

            ####################
            #Train the Model
            ####################

            if model_name == "LSTM_autoencoder":
                history = model.fit(
                        input_train, 
                        output_train, 
                        batch_size = batch_size,
                        epochs = training_epochs, 
                        validation_data = (input_valid, output_valid), 
                        verbose = 1,
                        callbacks = [callback_early_stopping, callback_checkpoint],
                        shuffle = True
                        )
            else:
                history = model.fit(
                        [train_X[:, :, 0].reshape(train_X.shape[0], train_X.shape[1], 1),train_X[:, :, 1].reshape(train_X.shape[0], train_X.shape[1], 1) ] , 
                        train_Y, 
                        batch_size = batch_size,
                        epochs = training_epochs, 
                        validation_data = ([valid_X[:, :, 0].reshape(valid_X.shape[0], valid_X.shape[1], 1),valid_X[:, :, 1].reshape(valid_X.shape[0], valid_X.shape[1], 1) ]  , valid_Y), 
                        verbose = 1,
                        callbacks = [callback_early_stopping, callback_checkpoint],
                        shuffle = True
                        )

            if flag_tuning == False:
                model.save('../models/{}/{}.h5'.format(model_name, model_name))

  
    #Load Best Checkpoint Model using Early Stopping 
    model = load_model('../models/{}/{}_best_model.h5'.format(model_name, model_name))

   
    plot_model(model, "{}_model.png".format(model_name), True, True)
    print(model.summary())


    #Plot Training and Validation Loss
    plot_loss_curve(history)




    ''''Inference stage '''

    if input_task == "3":  #If you are performing Forecasting
        if horizon > 1:
            #Predict the Y values for the given test set
            #predictions = predict_multi_timestep(model, test_X, horizon = horizon, model_name = model_name)  #Output shape (Batch_size, horizon, features)
            #plot_multistep_prediction(test_Y, predictions )

            if MIMO_output:

                if model_name == "LSTM_autoencoder":
                    #LSTM AUTOENCODER Predictor
                    decoder_model = build_prediction_model((1, train_Y.shape[-1]), units, cell_type)
                    predictions = predict_autoencoder(encoder_model, decoder_model, encoder_input_test)
                else:
                    predictions = predict_single_timestep(model, test_X)  #Output shape is (Batch_Size, n_features)
            else:
                #Predict the Y values for the given test set
                predictions = predict_multi_timestep(model, test_X, horizon = horizon, model_name = model_name)  #Output shape (Batch_size, horizon, features)
                #plot_multistep_prediction(test_Y, predictions )



           
            '''

                # invert predictions
            predictions = scaler.inverse_transform(predictions)
            test_Y = scaler.inverse_transform(test_Y)

            '''

            #Actual and Predicted values for Single electrode mutistep 
            true_elec = test_Y[:, :, 0]
            pred_elec = predictions[:, :, 0]

            #R value of a single electrode for all the time steps
            corr = list_correlation(true_elec, pred_elec)

            print("The value of correlation is for electrode 63 is {}". format(corr))

            
        else: 

            predictions = predict_single_timestep(model, test_X)  #Output shape is (Batch_Size, n_features)

            '''
            # invert predictions
            predictions = scaler.inverse_transform(predictions)
            test_Y = scaler.inverse_transform(test_Y)

            '''

            corr = list_correlation(predictions, test_Y)           #List of r value of all the the electrodes 

            print(corr)
                
        
        
        with open("corr_dat.json", "a") as write_file:
            write_file.write("\n")
            json.dump(corr, write_file)

        with open("experiment_log.json", "a") as write_file:
            json.dump({"Experiment_{}".format(experiment_no):corr }, write_file)
            write_file.write("\n")
       
    
       
    
    else: #Prediciting next time point of a single electrode or stimulus
        if horizon > 1:
                #Predict the Y values for the given test set
                #predictions = predict_multi_timestep(model, test_X, horizon = horizon, model_name = model_name)  #Output shape (Batch_size, horizon, features)
                #plot_multistep_prediction(test_Y, predictions )

                if MIMO_output:

                    if model_name == "LSTM_autoencoder":
                        #LSTM AUTOENCODER Predictor
                        decoder_model = build_prediction_model((1, train_Y.shape[-1]), units, cell_type)
                        predictions = predict_autoencoder(encoder_model, decoder_model, encoder_input_test)
                    else:
                        predictions = predict_single_timestep(model, test_X)  #Output shape is (Batch_Size, n_features)
                else:
                    #Predict the Y values for the given test set
                    predictions = predict_multi_timestep(model, test_X, horizon = horizon, model_name = model_name)  #Output shape (Batch_size, horizon, features)
                    #plot_multistep_prediction(test_Y, predictions )


                 #Actual and Predicted values for Single electrode mutistep 
                true_elec = test_Y[:, :, 0]
                pred_elec = predictions[:, :, 0]

                #R value of a single electrode for all the time steps
                corr = list_correlation(true_elec, pred_elec)

        else:
            predictions = predict_single_timestep(model, test_X)  #Output shape is (Batch_Size, n_features)

            '''
            # invert predictions
            predictions = scaler.inverse_transform(predictions)
            test_Y = scaler.inverse_transform(test_Y)

            '''

            corr = list_correlation(predictions, test_Y)           #List of r value of all the the electrodes 

            print(corr)
                

            
        with open("corr_dat.json", "a") as write_file:
            write_file.write("\n")
            json.dump(corr, write_file)

        with open("experiment_log.json", "a") as write_file:
            json.dump({"Experiment_{}".format(experiment_no):corr }, write_file)
            write_file.write("\n")

        
        if model_name == "LR":
             weights = np.array(model.get_weights())
             plot_weights(weights, pred, window)


    if flag_tuning == False:

        '''Save the predicted and True values in the numpy array'''
        savez_compressed('/root/EEG/models/{}/True.npz'.format(model_name), test_Y)
        savez_compressed('/root/EEG/models/{}/predicted.npz'.format(model_name), predictions)





