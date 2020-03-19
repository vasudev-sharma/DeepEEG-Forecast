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
  return format_1, trials



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



def split_data(data, window, trials, source_Y, source_X):

    #Split the trials
    #trials_train, trials_valid, trials_test = split_trials(trials)
    '''Hard coding of the results to produce consistent result'''
    trials_train = np.array([172, 125, 136,  99,  82,  31, 133,  44, 183, 184, 142, 121,  18,
        89, 141,  27, 107,  49,  68, 186,  70,  92, 109,   6, 147, 124,
       117, 161, 137,  39, 157, 159,   4,  23,  25, 145, 179, 118, 163,
       106,  69, 187,  76, 108, 188,  32, 178,  19,  26,  72, 168, 158,
        55,   8, 167,  11,  30,  59,  80,  95,  60, 148, 153,  45,  20,
       152,  73,  48,  36, 100, 185, 131, 138,   3,  13,  97, 126, 171,
       130,  54,   2,  50,  75,  83,  33, 174, 140,  79, 113, 146,  81,
        64,  63,  46, 170,  16, 173, 156,  90, 103, 144,  29,  58,  47,
       105, 189,  56,  34,  12, 165, 122, 119,  94,  42,  24,  37,  14,
        65,  93,  87, 154,  77, 166, 114, 112, 160, 164,  51, 139,  84,
       169,  85, 162,  88,  66, 155,  78,  28,   9,   1,  98, 132, 175,
       177, 115,  96, 111,  52,  21, 180,  61, 191, 143,  10])

    trials_valid = np.array([ 67,  15,  38,  22,   0,  74, 182, 151,  91,  43,  53, 123, 127,
        128, 149, 190, 134, 102, 181])

    trials_test = np.array([116,  57, 110,   7,  40, 176, 150,  41, 120, 135, 101,  71,  62,
            86, 129,  35, 104,   5,  17])

        

    #Extract Y
    y_train = extract_Y (data, window,  source_Y, trials_train)
    print ("y_train.shape = ", y_train.shape)

    y_valid = extract_Y (data, window, source_Y, trials_valid)
    print ("y_valid.shape = ", y_valid.shape)

    y_test = extract_Y (data, window, source_Y, trials_test)
    print ("y_test.shape = ", y_test.shape)

    #Extract X
    x_train = extract_X (data, window, source_X, trials_train)
    print ("x_train.shape = ", x_train.shape)

    x_valid = extract_X (data, window, source_X, trials_valid)
    print ("x_valid.shape = ", x_valid.shape)

    x_test = extract_X (data, window, source_X, trials_test)
    print ("x_test.shape = ", x_test.shape)



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
   
    
    x = np.array(np.split(x, len(source), axis = -1))
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

 
def get_info():

    eltmp = input ('Enter the electrode number:')
    electi = list(map(int, eltmp.split()))    #separation of the different responses and recovery in the form of a list of integers
    print (type(electi))
    print(electi)

    window = input ('Enter the number of stimuli:')
    window = int(window)

    n_channel = input('Enter  the chanel number for which you want your predicion: ')
    n_channel = list(map(int, n_channel.split()))


    relation = input('Please define what should be predicted (1 for EEG from stimulus or 2 for stimulus from EEG or 3 for EEG forecasting ):')

    if relation == '1':
        response = input("Do you want to embed information of EEG as well ? ( 1 for yes or 2 for no)")
        if response == "2":
            source_Y = electi   #retrieving the electrode number as a whole number - implies that there is only one electrode chosen in this direction
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

    return source_X, source_Y, window

def data():

    source_X, source_Y, window = get_info()
    
    #get data
    data, trials = get_data()

    return(split_data(data, window, trials, source_Y, source_X))








if __name__ == "__main__":

   data()

