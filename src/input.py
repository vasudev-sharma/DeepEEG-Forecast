import numpy as np
import scipy as sio
import matplotlib as plt
import numpy as np
import scipy.io as sio
rng = np.random

from scipy import stats
from utils import rolling_window



def get_data():

  # data import from matlab
  subject = sio.loadmat('../input/1filtered.mat')      #recovering matlab data in the form of a python dictionar
  format_1 = subject['data']          #in the dictionary, only the data key interests us
  print (format_1.shape)

  # shuffle trials
  (_, trial, _)= format_1.shape

  trials = np.arange(trial)
  np.random.shuffle(trials)

  #Z score
  format_1=stats.zscore(format_1, axis=2)
  return format_1



# extract Y for Muti Channel Prediction
def extract_Y (data, window, source, batch_trials):              #creation of a function to recover y - simplification of reading
   

    time_points = data.shape[-1]

    y = []
    for i in source:
        y_tmp=data[i, batch_trials, window:]     #recovery of Y in the form of a matrix of 154 * 840
        y_tmp=np.reshape(y_tmp, ((time_points-window)*len(batch_trials)))    #passage through the list of 129 360 values ​​(test 0, test 1, ... test 153)
        y_tmp=np.matrix(y_tmp)                           #1 * 129360 matrix conversion
        y_tmp=np.transpose(y_tmp)                        #transposition into a matrix of 129360 * 1, matrix equal to that of Matlab (necessary for the rest)
        y.append(y_tmp)
    return(np.hstack(y)) 
                                        #returns the content of y_tmp



def split_data(data, window, trials):

    #Split the trials
    trials_train, trials_valid, trials_test = split_trials(trials)

    #Extract Y
    y_train = extract_Y (data, window,  source_Y, trials_train)
    y_valid = extract_Y (data, window, source_Y, trials_valid)
    y_test = extract_Y (data, window, source_Y, trials_test)

    #Extract X
    x_train = extract_X (data, window, source_X, trials_train)
    x_valid = extract_X (data, window, source_X, trials_valid)
    x_test = extract_X (data, window, source_X, trials_test)


    return ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))



def extract_X (data,  window, source, batch_trials):                     #creation of a function to recover x - simplification of reading
   

    time_points = data.shape[-1]
    x = np.zeros((len(source),  len(batch_trials) * (time_points - window), window)) 

    for idx, i in enumerate(source):                                      #reading the source list -> reading each electrode number if flip
      x_tmp = []
      for j in batch_trials:
        tmp = rolling_window(data[i, j, :-1], window)
        x_tmp.append(tmp)
      x[idx] = np.vstack(x_tmp)

    x = np.hstack(x)
    x = np.array(np.split(x, len(source_X), axis = -1))
    x = np.moveaxis(x, 0, -1)
    return x



def split_trials(trials):
  #### separation of train tests / valid / test
  train_num = int(np.around(len(trials) * 0.8))
  valid_num = int(np.around(len(trials) * 0.1))

  trials_train = trials[0:train_num]
  trials_valid = trials[train_num:train_num+valid_num]
  trials_test = trials[train_num+valid_num:]

  return trials_train, trials_valid, trials_test

 

if __name__ == "main":

    a = get_data()
    print(a.shape)
    # parametres


    eltmp = input ('Enter the electrode number:')
    electi = list(map(int, eltmp.split()))    #separation of the different responses and recovery in the form of a list of integers
    print (type(electi))
    print(electi)

    window = input ('Enter the number of stimuli:')
    window = int(window)

    n_channel = input('Enter  the chanel number for which you want your predicion')
    n_channel = list(map(int, n_channel.split()))


    relation = input('Please define what should be predicted (1 for EEG from stimulus or 2 for stimulus from EEG or 3 for EEG forecasting ):')

    if relation == '1':
        response = input("Do you want to embed information of EEG as well ? ( 1 for yes or 2 for no)")
        if response == "2":
            source_Y = electi[0]    #retrieving the electrode number as a whole number - implies that there is only one electrode chosen in this direction
            source_X = [0]          #conversion of the stimuli line in the form of a list - necessary for the for loop: see below - extraction X
        else:
            source_Y = electi[0]
            source_X = [0] + electi[0]

    elif relation == '2':
        format_1 = np.flip(format_1,2)     # data inversion according to the time dimension - problem ????
        source_Y = 0
        source_X = electi
        
    elif relation == '3':
        response = input("Do you want to embed information of Stimuli as well ? ( 1 for yes or 2 for no)") 
        if response == "2":
            source_Y = n_channel
            source_X = electi
        else: 
            source_Y = n_channel
            source_X = electi + [0]       
    
