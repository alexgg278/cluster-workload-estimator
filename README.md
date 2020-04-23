# YAFS simulation time-series forecasting

The simulation implemented in this repo attempts to simulate a computing cluster using Yet Another Fog Simulator (YAFS). The goal is to model the cluster parameters such as memory usage by using time-series forecasting. With this purpose a simulation containing different Apps deployed on a cluster with different interconnected nodes is constructed.

## YAFS simulation scripts

The repository is divided in different tests. Each test have a goal of increasing complexity w.r.t the provious one. Most of the code from different tests is used in the following tests. However, some changes and enhacements are introduced. The goal of each test is described in the begining of the "testX.py" file of each test folder.

## Analysis scripts

The processing of the data and the analysis of the time-series is developed in a jupyter notebook file for each test "ts-analysis.ipynb". This files are only found from Test 4 since is when the analysis part started.
