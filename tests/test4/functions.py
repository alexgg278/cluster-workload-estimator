import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def data_from_node(df, node):
    """
    Takes as input the number of the desired node and dataframe
    This function returns a dataframe containing the data belonging to a node
    """
    return df[df['TOPO.dst'] == node]


def filter_time(df, time):
    """
    Takes as input the dataframe and the desired time to takes samples from
    Returns a dataframe containing samples where the time was found between the interval time_in and time_out
    That is returns the samples representing a message that was being executed in the passed time in the argument
    """
    return df[(df['time_in'] < time) & (time < df['time_out'])]


def ts_memory(df, node, param):
    """
    This functions takes as input a dataframe a node and the parameters
    Returns an array of time series with the used memory of the node at each time-step
    """
    ts_memory = []

    df = data_from_node(df, node)

    for time in tqdm(range(param.simulation_time)):
        memory = 0
        for idx, data in filter_time(df, time).iterrows():
            memory += data['memory']

        ts_memory.append([time, memory])

    return pd.DataFrame(data=ts_memory, columns=["time", "memory"])


def ts_memory_ds(df, node, param):
    """
    This functions takes as input a dataframe a node and the parameters
    Returns a dataframe of time series with the used memory of the node at each time-step
    """
    ts_memory = []

    df = data_from_node(df, node)

    for time in tqdm(list(np.arange(0, param.simulation_time, 0.1))):
        memory = 0
        for idx, data in filter_time(df, time).iterrows():
            memory += data['memory']

        ts_memory.append([time, memory])

    return pd.DataFrame(data=ts_memory, index=["time", "memory"])

def plot_ts(data, node):
    """
    This function take as input a time-series and plots it
    """
    plt.figure(figsize=(15, 6))
    plt.title('Time series')
    plt.xlabel('Time')
    plt.ylabel('Memory usage Node ' + str(node))
    plt.ylim(0, max(data)+100)
    plt.plot(data)
    plt.show()


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices[0]:indices[-1]+1].values, (history_size, 1)))
        labels.append(np.reshape(dataset[i: i + target_size].values, (target_size, 1)))
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx-', 'go-']
    time_steps = create_time_steps(plot_data[0].shape[0])

    future = range(delta)
    plt.figure(figsize=(14, 8))
    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i].flatten(), marker[i], markersize=5,
                   label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future[-1]+5)*2])
    plt.xlabel('Time-Step')
    plt.show()


def line_plot(line1, line2, label1=None, label2=None, title='Memory usage True vs Pred', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=3)
    ax.plot(line2, label=label2, linewidth=1)
    ax.set_ylabel('Memory usage', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)


def ms_val(y_val):
    y_val_plot = []
    for i in range(len(y_val)):
        if i == len(y_val) - 1:
            for j in range(len(y_val[i])):
                y_val_plot.append(y_val[i][j])
        else:
            y_val_plot.append(y_val[i][0])
    return np.array(y_val_plot)


def avg_pred(y_pred):
    avg_pred = []
    f_steps = y_pred.shape[1]
    w = np.zeros((f_steps))

    for idx, p in enumerate(y_pred):
        w += p
        if idx >= f_steps - 1:
            avg_pred.append(w[0] / f_steps)
        elif idx == len(y_pred) - 1:
            div = f_steps
            for j in range(len(y_pred[idx])):
                avg_pred.append(w[j] / div)
                div = div - 1
        else:
            avg_pred.append(w[0] / (idx + 1))
        w = np.roll(w, -1)
        w[f_steps - 1] = 0

    return np.array(avg_pred)