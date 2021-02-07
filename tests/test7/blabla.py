from parameters import Parameters

import pandas as pd
from functions import ts_memory_noise, ts_memory

# import parameters
param = Parameters()

# First we create the time-series and save them to a file
# Node to obtain time-series from
node = 3

# Storing the CSV into a DF
results = pd.read_csv("Results.csv")

# Obtaining time-series of desired nodes
ts = ts_memory(results, node, param)

ts = pd.concat([ts[0], ts[1]], axis = 1)

ts.to_pickle('ts_noise.pkl')