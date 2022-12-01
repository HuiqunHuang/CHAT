import h5py
import torch
import numpy as np
import pandas as pd

def loadDataAndDate(data_file_name):
    f = h5py.File(data_file_name, 'r')
    data = f['data'].value
    timeinfo = f['date'].value
    f.close()

    return data, timeinfo