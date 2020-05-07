import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf 
from metrics import compute_correlation
from sklearn.preprocessing import MinMaxScaler
import os 
import json

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def preprocess_data(data):
    #Reshape 3d array to 2d so it get's prrocessed by scikit learn
    channel, trial, time_points = data.shape[0], data.shape[1], data.shape[2]
    data = data.reshape(channel, -1)
    
    #Min max scaler for normalizing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(data)
    normalized = scaler.transform(data)
    
    #reshape the data to 3D array again
    normalized = normalized.reshape(channel, trial, time_points)


    return normalized, scaler


def inv_data(data, scaler):
    #Reshape 3d array to 2d so it get's prrocessed by scikit learn
    channel, trial, time_points = data.shape[0], data.shape[1], data.shape[2]
    data = data.reshape(channel, -1)
    
    #Retrieve the original data
    inversed = scaler.inverse_transform(data)

    #reshape the data to 3D array again
    original_data = inversed.reshape(channel, trial, time_points)
    return original_data

def plot_loss_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig("Train_Valid.png")
    plt.figure()
  

def show_data(data,channel, trial):
  # Plot EEG
  # plot of a single channel of EEG

  Time = np.linspace(0, 6.25, 1000)
  for i in range(channel):
    plt.plot(Time, data[i][trial])
    plt.savefig('1.png')
    plt.figure()



def sanity_check(pred, true):
    
    plt.plot(true[1550,:,0],'r')
    plt.plot(pred[1550,:,0],'b--')
    plt.savefig("../images/sanity_check_prediction_horizon.png")

    plt.figure()

    #sanity check: the "sequence" is present both in the first AND in the second dimension
    plt.plot(true[:500,0,0],'r')
    plt.plot(pred[:500,0,0],'b--')
    plt.savefig("../images/sanity_check_prediction_batch.png")
    #not so true when z_score_outputs is true, because the 2nd dimension was z-scored, the first wasn't
    #and, not so true with fake_data, i.e. when the signal is simple. The 'cheat' is only used for real EEG data.
    #plt.xlim(100,150)


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
  


  #Graphical Display to plot weights
def plot_weights(weights, electi, window):
 
        
    T = np.arange(0, 1, 1 / window)                       # creation of the time variable (on 1s) for the abscissa
    
    plt.plot(T, weights[0])
    plt.savefig("image.png") 
    plt.show()
    

def plot_multistep_prediction(true, pred):
    
    '''
    Plot multistep prediction after n horizon timesteps for all channels
    :pram true: Actual value that is test_Y            (Batch_size, Horizon, n_channels)
    :param pred: Predicted value in a recursive manner (Batch_size, Horizon, n_channels)
    '''

    ch_names = ['Stimulu','Fp1',   'AF7','AF3', 'F1','TP9','F5h','F7','FT7','FC5h','PO9','FC1','C1','C3',
    'C5','T7','TP7','CP5','CP3','CP1','P1','I1','P5h','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8',
     'AF4','AFz','Fz','F2','TP10','F6h','F8','FT8','FC6h','PO10','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2',
     'I2','P6h','P8',
     'P10', 'PO8','PO4','O2']

    time = np.arange(0, 1, 1 / 160)
    for j in range(true.shape[-1]):
        true_elec = true[:, :, j]
        pred_elec = pred[:, :, j]

        l = []
        for i in range(160):
            l.append(compute_correlation(true_elec[:,i], pred_elec[:, i]))

        name = "Channel_{}:-{}".format(j+1, ch_names[j+1])
        plt.xlabel('time points')
        plt.ylabel('r value')

        plt.plot(time, np.array(l), label = name)

        plt.legend(loc="upper right")
        plt.savefig("../images/"+name+".png")
        plt.figure()
    




    

def compare_plot_multistep_prediction(array, model_names, baseline):
    
    '''
    Plot multistep prediction after n horizon timesteps for all channels
    :pram true: Actual value that is test_Y            (Batch_size, Horizon, n_channels)
    :param pred: Predicted value in a recursive manner (Batch_size, Horizon, n_channels)
    '''

    ch_names = ['Stimulu','Fp1',   'AF7','AF3', 'F1','TP9','F5h','F7','FT7','FC5h','PO9','FC1','C1','C3',
    'C5','T7','TP7','CP5','CP3','CP1','P1','I1','P5h','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8',
     'AF4','AFz','Fz','F2','TP10','F6h','F8','FT8','FC6h','PO10','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2',
     'I2','P6h','P8',
     'P10', 'PO8','PO4','O2']

    time = np.arange(0, 1, 1 / 160)
    list_correlation_avg = []
    baseline_avg = []
 
    for j in range(64):
       
        baseline_elec = baseline[:,j]

    


        list_correlation_model = []
        for k in range(len(array)):
            list_correlation = []

            true_elec = array[k][0][:, :, j]
            pred_elec =array[k][1][:, :, j]  
            for i in range(160):

                list_correlation.append(compute_correlation(true_elec[:,i], pred_elec[:, i]))
            list_correlation_model.append(list_correlation)


        name = "Channel_{}:-{}".format(j+1, ch_names[j+1])
        plt.xlabel('time points')
        plt.ylabel('r value')

        print(len(list_correlation_model))
        print(len(list_correlation_model[0]))

        for i in range(len(model_names)):
            plt.plot(time, np.array(list_correlation_model[i]), label = name + ' '+ model_names[i])
        plt.plot(time, baseline_elec, label = name+ ' Baseline')


        baseline_avg.append(baseline_elec)
        list_correlation_avg.append(list_correlation_model)


        plt.legend(loc="upper right")
        plt.savefig("../images/"+name+".png")
        plt.figure()
        
    plt.xlabel('time points')
    plt.ylabel('r value ')
 
    print(np.array(baseline_avg).shape)

    for i in range(len(model_names)):
        plt.plot(time, np.array(list_correlation_avg)[:,i,:].mean(axis = 0), label = ' ' + model_names[i])
    plt.plot(time, np.array(baseline_avg).mean(axis = 0), label =' Baseline')

    plt.legend(loc="upper right")
    plt.savefig("../images/"+"Average_over_64_channels"+".png")
    plt.figure()


def compare_models():
    color_list = ['r', 'b', 'g', 'c', 'm', 'y']
    model_names_list = ['LSTM_prediction', 'LSTM_prediction_from_Stimulus', 'LSTM_prediction_with_Stimulus', 'EEG_Predictions_from_Stimulus_CNN', "Combined_model", "bbbbbbbb", "cc"]
    with open("models.json", "r") as read_file:
            data = read_file.readlines() 
            plt.xlabel('time points')
            plt.ylabel('r value')
            time = np.arange(0, 160)
            for i in range(len(data)):
                plt.plot(time, json.loads(data[i]), color_list[i], label = model_names_list[i])
                plt.legend()
            
            plt.savefig("../images/models_comparison.png")
            plt.figure()    


'''
def compare_plot_multistep_prediction(true, pred, true1, pred1):
    
    
    Plot multistep prediction after n horizon timesteps for all channels
    :pram true: Actual value that is test_Y            (Batch_size, Horizon, n_channels)
    :param pred: Predicted value in a recursive manner (Batch_size, Horizon, n_channels)
    

    ch_names = ['Stimulu','Fp1',   'AF7','AF3', 'F1','TP9','F5h','F7','FT7','FC5h','PO9','FC1','C1','C3',
    'C5','T7','TP7','CP5','CP3','CP1','P1','I1','P5h','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8',
     'AF4','AFz','Fz','F2','TP10','F6h','F8','FT8','FC6h','PO10','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2',
     'I2','P6h','P8',
     'P10', 'PO8','PO4','O2']

    time = np.arange(0, 1, 1 / 160)
    for j in range(true.shape[-1]):
        true_elec = true[:, :, j]
        pred_elec = pred[:, :, j]

        true_elec1 = true1[:, :, j]
        pred_elec1 = pred1[:, :, j]


        l = []
        k = []
        for i in range(160):
            l.append(compute_correlation(true_elec[:,i], pred_elec[:, i]))
            k.append(compute_correlation(true_elec1[:,i], pred_elec1[:, i]))

        name = "Channel_{}:-{}".format(j+1, ch_names[j+1])
        plt.xlabel('time points')
        plt.ylabel('r value')

        plt.plot(time, np.array(l), label = name)
        plt.plot(time, np.array(k), label = name)

        plt.legend(loc="upper right")
        plt.savefig("../images/"+name+".png")
        plt.figure()
    
    
    '''