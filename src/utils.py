import numpy as np
import matplotlib.pyplot as plt

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
    plt.figure()
