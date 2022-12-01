# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import _pickle as pickle
import numpy as np
import math
import tensorflow as tf

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from DataProcessing.PrepareDataForModel import loadData
from FileOperation import CSVWrite
from Model.CHAT_Model import categoricalTimeSeriesDataPredictionModel
from config import Config
import metrics as metrics
from matplotlib import pyplot
import random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1337
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATAPATH = Config().DATAPATH
nb_epoch_cont = 300
batch_size = 96
T = 24 # number of time interval within a day, if one time interval is 1 hour, then T is 24
lr = 0.002 # learning rate
len_closeness = 5
days_test = 10
days_val = 5
len_test = T * days_test
len_val = T * days_val
category_num = 8
feature_num, height, width = 1, 4, 3 # height and width are the the rows and columns of the heatmap, respectively
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

def build_model():
    c_conf = (len_closeness, feature_num, height, width) if len_closeness > 0 else None

    model = categoricalTimeSeriesDataPredictionModel(c_conf=c_conf, batch_size=batch_size, out_height=height, out_width=width, category_num=category_num)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    return model


def main(datatype, category_data_file_name, heatmap_data_file_name, city):
    print("loading data...")
    X_data_train, X_data_val, X_data_test, Y_data_train, Y_data_val, Y_data_test, c_X_train, \
    c_X_val, c_X_test, c_Y_train, c_Y_val, c_Y_test, heatmap_X_train, heatmap_X_val, heatmap_X_test, \
    heatmap_Y_train, heatmap_Y_val, heatmap_Y_test = loadData(datatype, category_data_file_name,
                                                                        heatmap_data_file_name, T=T,
                                                                        len_closeness=len_closeness,
                                                                        len_test=len_test, len_val=len_val)

    print('=' * 10)
    print("compiling model...")
    model = build_model()
    hyperparams_name = 'c{}.lr{}'.format(
        len_closeness, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
    print("training model...")
    early_stopping = EarlyStopping(monitor='val_rmse', patience=1000, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_data_train, c_Y_train,
                        epochs=nb_epoch_cont,
                        batch_size=batch_size,
                        validation_data=(X_data_val, c_Y_val),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    total_mae = 0
    total_mse = 0
    total_msle = 0
    total_mape = 0
    length = 0
    total_ground_truth = 0
    y_predict = model.predict(X_data_test)
    total_dis = 0
    mean_gro = np.mean(np.array(c_Y_test))
    print(np.array(y_predict).shape)
    print(np.array(c_Y_test).shape)

    for i in range(0, len(y_predict)):
        for j in range(0, len(y_predict[i])):
            ab = abs(c_Y_test[i][j] - y_predict[i][j])
            cc = abs(c_Y_test[i][j] - mean_gro)
            total_dis += (cc * cc)
            total_ground_truth += c_Y_test[i][j]
            total_mae += ab
            total_mse += (ab * ab)
            aa = math.log(c_Y_test[i][j] + 1, 2) - math.log(y_predict[i][j] + 1, 2)
            total_msle += (aa * aa)
            if c_Y_test[i][j] == 0:
                bb = 0
            else:
                bb = abs((c_Y_test[i][j] - y_predict[i][j]) / c_Y_test[i][j])
            total_mape += bb
            length += 1

    MAE = total_mae / length
    MSE = total_mse / length
    MSLE = total_msle / length
    MAPE = total_mape / length
    ER = total_mae / total_ground_truth

    print("City: " + city)
    print("MAE: " + str(MAE))
    print("MSE: " + str(MSE))
    print("RMSE: " + str(math.sqrt(MSE)))
    print("MSLE: " + str(MSLE))
    print("MAPE: " + str(MAPE))
    print("Error Rate TOTAL: " + str(ER))


    pyplot.plot(history.history['loss'])
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("Training Loss", fontsize=18)
    pyplot.legend()
    pyplot.show()

    pyplot.plot(history.history['rmse'])
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("Training RMSE", fontsize=18)
    pyplot.legend()
    pyplot.show()

    pyplot.plot(history.history['val_loss'])
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("Validation Loss", fontsize=18)
    pyplot.legend()
    pyplot.show()

    pyplot.plot(history.history['val_rmse'])
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("Validation RMSE", fontsize=18)
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    with tf.device("gpu:0"):
        main("chicago bike 2021",
             "Chicago_BikePickUps_20210101_20211231_8Clusters.h5",
             "Chicago_BikePickUPs_HeatMap_4_3_20210101_20211231_8Clusters.h5",
             "Chicago")


