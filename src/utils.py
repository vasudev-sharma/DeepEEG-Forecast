import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf 
from tensorflow import keras as K
from metrics import compute_correlation
import os 

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def show_data(data,channel, trial):
  # Plot EEG
  # plot of a single channel of EEG

  Time = np.linspace(0, 6.25, 1000)
  for i in range(channel):
    plt.plot(Time, data[i][trial])
    plt.savefig('1.png')
    plt.figure()



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
 
        
    T = np.arange(0, 0.00625*window, 0.00625)                       # creation of the time variable (on 1s) for the abscissa
    
    nbr_elct = len(electi)
    (Lo_W, la_W) = weights[0].shape
    Lo_W = int(Lo_W)                                              # number of lines: inputs (160 * nbr_electrodes, for example)
    la_W = int(la_W)                                              # number of columns: number of neurons in the layer
    
    z=0
    while z < nbr_elct:                                           # for each electrode
        z_1=0 
        while z_1 < la_W:                                         # for each neuron of the W layer [1]
            W_tmp = tf.slice(K.constant(weights[0]), [0, z_1], [Lo_W, 1])           # slice: starting value [line 0, column of the neuron], dimensions of the section [160 * nbr_electrodes lines, 1 column])
            W_tmp = tf.slice(W_tmp, [z*window, 0], [window, 1])       # slice: starting value [first value of the new electrode, column 0], dimensions of the section [160 lines, 1 column]
            print (W_tmp.shape)
            
            plt.plot(T, K.eval(W_tmp), label= ("neurone_", z_1, "layer_1, electrode_", electi[z]))
            z_1 = z_1+1
            
        z = z+1
        
    plt.legend()    
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
    




    

def compare_plot_multistep_prediction(true, pred, true1, pred1, baseline):
    
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
    l_avg = []
    k_avg = []
    baseline_avg = []
    for j in range(true.shape[-1]):
        true_elec = true[:, :, j]
        pred_elec = pred[:, :, j]
        baseline_elec = baseline[:,j]

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

        plt.plot(time, np.array(l), label = name + ' LR Prediction')
        plt.plot(time, np.array(k), label = name+ ' LSTM Prediction')
        plt.plot(time, baseline_elec, label = name+ ' Baseline')

        l_avg.append(np.array(l))
        k_avg.append(np.array(k))
        baseline_avg.append(baseline_elec)


        plt.legend(loc="upper right")
        plt.savefig("../images/"+name+".png")
        plt.figure()
    
    plt.xlabel('time points')
    plt.ylabel('r value ')
    print(np.array(l_avg).shape)
    print(np.array(k_avg).shape)
    print(np.array(baseline_avg).shape)


    plt.plot(time, np.array(l_avg).mean(axis = 0), label = ' LR Prediction')
    plt.plot(time, np.array(k_avg).mean(axis = 0), label =' LSTM Prediction')
    plt.plot(time, np.array(baseline_avg).mean(axis = 0), label =' Baseline')

    plt.legend(loc="upper right")
    plt.savefig("../images/"+"Average_over_64_channels"+".png")
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