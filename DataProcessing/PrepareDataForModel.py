import numpy as np
import pandas as pd
import os
from FileOperation.LoadH5 import loadDataAndDate
from config import Config
DATAPATH = Config().DATAPATH
import openturns as ot
ot.Log.Show(ot.Log.NONE)
ot.RandomGenerator.SetSeed(0)

def loadData(datatype, category_data_file_name, heatmap_data_file_name, T=24, len_closeness=4, len_test=None, len_val=None):
    assert (len_closeness > 0)

    if datatype == "chicago bike 2021":
        target_data, timeinfo = loadDataAndDate(
            os.path.join(DATAPATH, 'Data\\', category_data_file_name))
        heatmap_data, timeinfo = loadDataAndDate(
            os.path.join(DATAPATH, 'Data\\', heatmap_data_file_name))

    if datatype == "chicago bike 2021":
        # using the check-in data from 2021-03-10 to 2021-06-10, normal
        # target_data = target_data[T * 68:T * 161]
        # heatmap_data = heatmap_data[T * 68:T * 161]
        # using the check-in data from 2021-07-31 to 2021-10-31, heavy rain and strong wind
        target_data = target_data[T * 211:T * 304]
        heatmap_data = heatmap_data[T * 211:T * 304]
        # using the check-in data from 2021-08-31 to 2021-11-30, Thanks Giving Day
        # target_data = target_data[T * 242:T * 336]
        # heatmap_data = heatmap_data[T * 242:T * 336]

    c_X, c_Y = [], []
    heatmap_X, heatmap_Y = [], []

    for i in range(len_closeness, len(target_data)):
        c_X.append(np.asarray([np.vstack([target_data[j] for j in range(i - len_closeness, i)])]))
        c_Y.append(np.asarray(target_data[i]))

        heatmap_X.append(np.asarray([np.vstack(
            [heatmap_data[j] for j in range(i - len_closeness, i)])]))
        heatmap_Y.append(np.asarray(heatmap_data[i]))

    c_X = np.vstack(c_X)
    c_Y = np.vstack(c_Y)
    heatmap_X = np.vstack(heatmap_X)
    heatmap_Y = np.vstack(heatmap_Y)

    print("closeness_X shape: " + str(c_X.shape))
    print("closeness_Y shape: " + str(c_Y.shape))
    print("heatmap_X shape: " + str(heatmap_X.shape))
    print("heatmap_Y shape: " + str(heatmap_Y.shape))

    c_X_train, c_X_val, c_X_test = c_X[:-(len_test + len_val)], c_X[-(len_test + len_val):-len_test], c_X[-len_test:]
    c_Y_train, c_Y_val, c_Y_test = c_Y[:-(len_test + len_val)], c_Y[-(len_test + len_val):-len_test], c_Y[-len_test:]
    heatmap_X_train, heatmap_X_val, heatmap_X_test = heatmap_X[:-(len_test + len_val)], heatmap_X[-(len_test + len_val):-len_test], heatmap_X[-len_test:]
    heatmap_Y_train, heatmap_Y_val, heatmap_Y_test = heatmap_Y[:-(len_test + len_val)], heatmap_Y[-(len_test + len_val):-len_test], heatmap_Y[-len_test:]

    c_X_train = np.array(c_X_train)
    c_X_val = np.array(c_X_val)
    c_X_test = np.array(c_X_test)
    c_Y_train = np.array(c_Y_train)
    c_Y_val = np.array(c_Y_val)
    c_Y_test = np.array(c_Y_test)
    heatmap_X_train = np.array(heatmap_X_train)
    heatmap_X_val = np.array(heatmap_X_val)
    heatmap_X_test = np.array(heatmap_X_test)
    heatmap_Y_train = np.array(heatmap_Y_train)
    heatmap_Y_val = np.array(heatmap_Y_val)
    heatmap_Y_test = np.array(heatmap_Y_test)

    print("c_X_train shape: " + str(c_X_train.shape))
    print("c_X_val shape: " + str(c_X_val.shape))
    print("c_X_test shape: " + str(c_X_test.shape))
    print("")
    print("c_Y_train shape: " + str(c_Y_train.shape))
    print("c_Y_val shape: " + str(c_Y_val.shape))
    print("c_Y_test shape: " + str(c_Y_test.shape))
    print("")
    print("heatmap_X_train shape: " + str(heatmap_X_train.shape))
    print("heatmap_X_val shape: " + str(heatmap_X_val.shape))
    print("heatmap_X_test shape: " + str(heatmap_X_test.shape))
    print("")
    print("heatmap_Y_train shape: " + str(heatmap_Y_train.shape))
    print("heatmap_Y_val shape: " + str(heatmap_Y_val.shape))
    print("heatmap_Y_test shape: " + str(heatmap_Y_test.shape))
    print("")

    X_data_train = heatmap_X_train
    X_data_val = heatmap_X_val
    X_data_test = heatmap_X_test

    Y_data_train = c_Y_train
    Y_data_val = c_Y_val
    Y_data_test = c_Y_test


    return X_data_train, X_data_val, X_data_test, Y_data_train, Y_data_val, Y_data_test, c_X_train, \
           c_X_val, c_X_test, c_Y_train, c_Y_val, c_Y_test, heatmap_X_train, heatmap_X_val, heatmap_X_test, \
           heatmap_Y_train, heatmap_Y_val, heatmap_Y_test
