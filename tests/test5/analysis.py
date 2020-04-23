import pandas as pd
import tensorflow as tf

from parameters import Parameters
from functions import ts_memory, univariate_data, ms_val, avg_pred, line_plot

# import parameters
param = Parameters()
node = 3

"""
# First we create the time-series and save them to a file
# Node to obtain time-series from
node = 3

# Storing the CSV into a DF
results = pd.read_csv("Results.csv")

# Obtain time series of memory usage in node 3
ts = ts_memory(results, node, param)

# Save them to disk
ts.to_pickle('ts.pkl')
"""
# Use the already saved time-series by loading them from file
ts = pd.read_pickle('ts.pkl')

# Plot time-series
# plot_ts(ts["memory"], node)

# Plot time-series
# plot_ts(ts["memory"][:2000], node)

# Set seed for reproducibility
tf.random.set_seed(13)

# Take data
ts_data = ts['memory']
ts_data.index = ts['time']

# We will use 70% percent of data to train
TRAIN_SPLIT = 0.7
TRAIN_SPLIT = int(TRAIN_SPLIT*ts_data.size)

# Normalize data
ts_data_mean = ts_data[:TRAIN_SPLIT].mean()
ts_data_std = ts_data[:TRAIN_SPLIT].std()
ts_data = (ts_data-ts_data_mean) / ts_data_std

# Get the training and validation data and targets
univariate_past_history = 150
univariate_future_target = 5

x_train, y_train = univariate_data(ts_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val, y_val = univariate_data(ts_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

# show_plot([x_train[0], y_train[0]], univariate_future_target, 'Sample Example')

BATCH_SIZE = 100
# BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=x_train.shape[-2:]),
    tf.keras.layers.Dense(univariate_future_target)
])

lstm_model.compile(optimizer='adam', loss='mae')

EVALUATION_INTERVAL = 200
EPOCHS = 10

lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
"""
for x, y in val_univariate:
    plot = []
    for i in range(len(x)):
        plot.append(show_plot([x[i].numpy(), y[i].numpy(),
                               lstm_model.predict(x)[i]], univariate_future_target, 'Simple LSTM model'))
"""

y_pred = lstm_model.predict(x_val)

y_val_plot = ms_val(y_val)
y_pred = avg_pred(y_pred)

line_plot(y_val_plot.flatten(), y_pred.flatten(), 'True', 'Pred')


