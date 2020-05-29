import os

import json
import tensorflow
from tensorflow.keras import optimizers
from input import data, teacher_forcing
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from predict import predict_single_timestep, predict_multi_timestep, predict_autoencoder
from models import get_model, build_prediction_model
from metrics import compute_correlation, list_correlation, mean_squared_loss, cosine_loss
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import plot_multistep_prediction, plot_loss_curve, sanity_check, plot_r_horizon
from numpy import savez_compressed
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import numpy as np
from tqdm import tqdm
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import sys
from utils import plot_weights

import random 
import string


############################
#Set the seed to produce consistent results
###########################
'''
tensorflow.random.set_seed(0)
random.seed(0)
np.random.seed(0)
'''

# This is secret and shouldn't be checked into version control
os.environ['WANDB_API_KEY']='202040aaac395bbf5a4a47d433a5335b74b7fb0e'
os.environ['WANDB_MODE'] = 'dryrun'
pred = os.environ["pred"]
stimulus = os.environ["stimulus"]
input_task = os.environ["input_task"]
model_name = os.environ["model_name"]
horizon = os.environ["horizon"]
training = os.environ["training"]
MIMO_output = os.environ["MIMO_output"]
experiment_no=os.environ["experiment_no"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 




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




    #Read the parameters of the model
    with open("../config/{}/parameters.json".format(model_name), "r") as param_file:
        parameters = json.load(param_file)

    #Parameters of model
    training_epochs = parameters["training_epochs"]
     
    batch_size = parameters["batch_size"]
    
    units = parameters["units"]
    learning_rate = parameters["learning_rate"]
    if(model_name == "LSTM" or model_name=="LSTM_hp"  or model_name=="conv_LSTM" or model_name == "combined_model" or model_name =="ES_RNN"):
        cell_type = parameters["cell_type"]
    if model_name == "LSTM_autoencoder":
        cell_type = parameters["cell_type"]
        teacher_force = bool(parameters["teacher_force"])


    #######
    #LOG Congif Parameters
    #######
    
    wandb.init(config=parameters, project= "input_task_" + input_task  + "_" +  "stimulus_" + stimulus + "_" + "Prediction_" + '3, 13, 18, 27, 30, 32, 36, 37, 47, 50, 55, 64' + "_" + "Model_name_" + model_name + "_" + "Horizon_" + str(horizon) + "_" + "Output_type_" + MIMO_output )
    

    if model_name == "LSTM_autoencoder":
      encoder_input_train, decoder_input_train, decoder_target_train = teacher_forcing(train_X, train_Y)
      encoder_input_valid, decoder_input_valid,  decoder_target_valid = teacher_forcing(valid_X, valid_Y)
      encoder_input_test, decoder_input_test,  decoder_target_test = teacher_forcing(test_X, test_Y)


      if not teacher_force: #Set to False for not teacher forcing, set to True for Teacher_forcing 
        print("Teacher force is used")  
        '''
        decoder_input_train = decoder_input_train[:, :1, :1]
        decoder_input_valid = decoder_input_valid[:, :1, :1]
        decoder_input_test = decoder_input_test[:, :1, :1]
        '''
        input_train = [encoder_input_train, decoder_input_train]
        input_valid = [encoder_input_valid, decoder_input_valid]
        input_test = [encoder_input_test, decoder_input_test]


      else:
        input_train = encoder_input_train
        input_valid = encoder_input_valid
        input_test = encoder_input_test



      output_train = decoder_target_train
      output_valid = decoder_target_valid
      print("Shape of encoder_input_train, decoder_input_train, decoder_target_train is  ",  encoder_input_train.shape, decoder_input_train.shape, decoder_target_train.shape)
      print("Shape of encoder_input_valid, decoder_input_valid,  decoder_target_valid is ", encoder_input_valid.shape, decoder_input_valid.shape,  decoder_target_valid.shape)  
     



    
    if training: 



            if not os.path.exists("../models/{}".format(model_name,)):
                os.mkdir("../models/{}".format(model_name))

            '''
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                    patience=10, min_lr=0.00001)
            '''
            ####################################
            # Callbacks    
            ###################################
            
            callback_early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=20, mode = 'min')
            callback_checkpoint = ModelCheckpoint("../models/{}/{}_best_model.h5".format(model_name, model_name), mode = "min", monitor='val_loss', save_best_only=True, verbose = 1)
            
            
            wandb.config.loss = "MSE"
            wandb.config.optimizer = "Adam"

            ############################
            #Compile the model 
            ############################
            model = get_model()[model_name]
            if model_name == "LSTM" or model_name=="conv_LSTM" or model_name == "combined_model":
                model = model(train_X.shape, units, train_Y.shape[-1], cell_type, learning_rate,  wandb.config.loss, wandb.config.optimizer)
            elif model_name =="LSTM_autoencoder":
                model, encoder_model = model(train_X.shape, units, train_Y.shape[-1], cell_type, learning_rate, teacher_force,  wandb.config.loss, wandb.config.optimizer)
            elif model_name == "CNN" or model_name =="CNN_cross":
                model = model(train_X.shape, train_Y.shape[-1], learning_rate,  wandb.config.loss, wandb.config.optimizer)
            elif model_name =="LR":
                model = model(train_X.shape, learning_rate,  wandb.config.loss, wandb.config.optimizer)
            elif model_name =="ES_RNN":
                model = model(train_X.shape, units, train_Y.shape[-1], cell_type, learning_rate,  wandb.config.loss, wandb.config.optimizer, batch_size)
            else:
                model = model 
            



            if flag_tuning == True:
                random_string = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])

                tuner_search=RandomSearch(model,
                                        objective='val_loss',
                                        max_trials=200,project_name= "experiment/"+ random_string,
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
                
                training_epochs = 20


            print(model.summary())
            # Fit the model with the Data
            plot_model(model, "../images/{}_model.png".format(model_name), True, True)

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
                        callbacks = [ callback_early_stopping, callback_checkpoint, WandbCallback()],
                        shuffle = True
                        )
                    #callback_early_stopping, callback_checkpoint,

            elif model_name == "combined_model":

            
                history = model.fit(
                        [train_X[:, :, 0].reshape(train_X.shape[0], train_X.shape[1], 1),train_X[:, :, 1].reshape(train_X.shape[0], train_X.shape[1], 1) ] , 
                        train_Y, 
                        batch_size = batch_size,
                        epochs = training_epochs, 
                        validation_data = ([valid_X[:, :, 0].reshape(valid_X.shape[0], valid_X.shape[1], 1),valid_X[:, :, 1].reshape(valid_X.shape[0], valid_X.shape[1], 1) ]  , valid_Y), 
                        verbose = 1,
                        callbacks = [callback_early_stopping, callback_checkpoint, WandbCallback()],
                        shuffle = True
                        )
            

            else:
                history = model.fit(
                        train_X, 
                        train_Y, 
                        batch_size = batch_size,
                        epochs = training_epochs, 
                        validation_data = (valid_X  , valid_Y), 
                        verbose = 1,
                        callbacks = [callback_early_stopping, callback_checkpoint, WandbCallback()],
                        shuffle = True
                        )
            
            



            if flag_tuning == False:
                model.save('../models/{}/{}.h5'.format(model_name, model_name))
        
            #Plot Training and Validation Loss
            plot_loss_curve(history)


    
    #Load Best Checkpoint Model using Early Stopping 
    if not wandb.config.loss == "MSE":
        print("Cosine loss is used")
        model = load_model('../models/{}/{}_best_model.h5'.format(model_name, model_name), custom_objects={"cosine_loss":cosine_loss} )
    else:
        print("MSE loss is used")
        model = load_model('../models/{}/{}_best_model.h5'.format(model_name, model_name))
    
    plot_model(model, "../images/{}_model.png".format(model_name), True, True)
    print(model.summary())




    ####################################
    #Inference stage
    ####################################


    if horizon > 1:
        #Predict the Y values for the given test set
        #predictions = predict_multi_timestep(model, test_X, horizon = horizon, model_name = model_name)  #Output shape (Batch_size, horizon, features)
        #plot_multistep_prediction(test_Y, predictions )

        if MIMO_output:

            if model_name == "LSTM_autoencoder":
                
                predictions = predict_single_timestep(model, input_test)  #Output shape is (Batch_Size, n_features)
                '''
                #LSTM AUTOENCODER Predictor
                decoder_model = build_prediction_model((1, train_Y.shape[-1]), units, cell_type)
                predictions = predict_autoencoder(encoder_model, decoder_model, encoder_input_test)
                '''
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

        print("Shape of true  is", test_Y.shape)
        print("Shape of pred  is ", predictions.shape)

        if len(test_Y.shape) == 3 and len(predictions.shape) == 3:
            #Actual and Predicted values for Single electrode mutistep 
            pass
            #true_elec = test_Y[:, :, 0]
            #pred_elec = predictions[:, :, 0]
            true_elec = test_Y
            pred_elec = predictions

        else: 
            true_elec = test_Y
            pred_elec = predictions

        print("Shape of true elec is", true_elec.shape)
        print("Shape of pred elec is ", pred_elec.shape)

        #R value of a single electrode for all the time steps
        corr = list_correlation(true_elec, pred_elec)

        #plot_r_horizon(corr)
         
       
        for  i in range(12):
            #R value of a single electrode for all the time steps
            corr = list_correlation(true_elec[:,:, i], pred_elec[:, :, i])
            print("The value of correlation is for electrode is {}". format(corr))

        
    else: 

        predictions = predict_single_timestep(model, test_X)  #Output shape is (Batch_Size, n_features)

        '''
        # invert predictions
        predictions = scaler.inverse_transform(predictions)
        test_Y = scaler.inverse_transform(test_Y)

        '''

        corr = list_correlation(predictions, test_Y)           #List of r value of all the the electrodes 

        print(corr)
        plot_r_horizon(corr)

    
    
    with open("corr_dat.json", "a") as write_file:
        write_file.write("\n")
        json.dump(corr, write_file)

    with open("experiment_log.json", "a") as write_file:
        json.dump({"Experiment_{}".format(experiment_no):corr }, write_file)
        write_file.write("\n")
    

    

        
        if model_name == "LR":
            if not multivariate:
             weights = np.array(model.get_weights())
             plot_weights(weights, pred, window)
        
    
    #Perform sanity check to check the model is performing the correct prediction over future time steps horzizons and over certi
    sanity_check(test_Y, predictions, MIMO_output)

    '''
    #Log the images

    wandb.log({
    "sanity_check_prediction_batch":  wandb.Image("../images/sanity_check_prediction_batch.png")})

    wandb.log({"sanity_check_prediction_horizon":  wandb.Image("../images/sanity_check_prediction_horizon.png")})
    
    for i in range(160):
        wandb.log({"corr_value": corr[i], "global_step" :i})


    wandb.log({"corr_list": corr})


    time = np.arange(0, 160)
    plt.plot(time, corr)
    wandb.log({"corr_plot": plt})
    plt.figure()

    wandb.log({
    "Prediction_horizon":  wandb.Image("./Prediction.png")})

    '''
   






    if flag_tuning == False:

        '''Save the predicted and True values in the numpy array'''
        savez_compressed('/root/EEG/models/{}/True.npz'.format(model_name), test_Y)
        savez_compressed('/root/EEG/models/{}/predicted.npz'.format(model_name), predictions)





