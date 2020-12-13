from utils import plot_multistep_prediction, compare_plot_multistep_prediction, compare_models
from metrics import list_correlation
from tqdm import tqdm
from predict import baseline
import numpy as np
from numpy import savez_compressed
import matplotlib.pyplot as plt
import wandb

# Your custom arguments defined here


if __name__ == "__main__":

    '''
    pred = -1
    horizon = 160
    train,  valid, test = data(int(pred), input_task= "3", stimulus= "2", horizon = horizon,  split = True , multivariate = True)
    test_X, test_Y = test
    baseline = baseline(test_X, test_Y)

    
    
    
    model_names = ["LR", "GRU", "RNN", "LSTM", "LSTM_autoencoder"]
    array_models = []
    for i in range(len(model_names)):
        true = np.load("../models/{}/True.npz".format(model_names[i]))
        pred = np.load("../models/{}/predicted.npz".format(model_names[i]))
        print("Name of the model is ", model_names[i])
        print("Shape of the actual test set is - ", true['arr_0'].shape)
        print("Shape of the predicted test set is - ", pred['arr_0'].shape)
        array_models.append((true['arr_0'],pred['arr_0']))


    baseline = np.load('../models/baseline/baseline_all_channels.npz')




    print("Baseline")
    print("Shape of Baseline model is - ", baseline['arr_0'].shape)
    
    compare_plot_multistep_prediction(array_models, model_names, baseline['arr_0'])


    print("hello")

    

    compare_models()

    '''
    args = dict(
       
    )

    wandb.init(config=args, project="my-project")
    wandb.config["more"] = "custom"

    for i in range(2):
        # Do some machine learning
        epoch, loss, val_loss = 1, 2, 3
        # Framework agnostic / custom metrics
        wandb.log({"epoch": epoch, "loss": loss, "val_loss": val_loss})
    
    wandb.log({
        "image":  wandb.Image("../images/models_comparison.png")})

            
