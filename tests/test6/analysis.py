import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from parameters import Parameters
from functions import ts_memory, show_plot, plot_ts, multivariate_data, ms_val, avg_pred, line_plot

nodes = [2, 3]

# import parameters
param = Parameters()

# First we create the time-series and save them to a file
# Node to obtain time-series from
nodes = [2, 3]

ts = pd.read_pickle('ts.pkl')

# Plot time-series
# plot_ts(ts["memory"], node)

# Plot time-series
for node in nodes:
    plot_ts(ts['memory_'+str(node)][2000:5000], node)

features = ['memory_'+str(node) for node in nodes]

# Set seed for reproducibility
tf.random.set_seed(13)

# Take data
ts_data = ts[features]
x = ts_data.shape
# We will use 70% percent of data to train
TRAIN_SPLIT = 0.7
TRAIN_SPLIT = int(TRAIN_SPLIT*ts_data.shape[0])

# Normalize data
ts_data_mean = ts_data[:TRAIN_SPLIT].mean(axis=0)
ts_data_std = ts_data[:TRAIN_SPLIT].std(axis=0)
ts_data = (ts_data-ts_data_mean) / ts_data_std

past_history = 150
future_target = 1
STEP = 1

x_train_single, y_train_single = multivariate_data(ts_data, ts_data['memory_3'], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(ts_data, ts_data['memory_3'],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)