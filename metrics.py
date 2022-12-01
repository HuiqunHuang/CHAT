from tensorflow.python.keras import backend as K
import tensorflow as tf


@tf.function
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

@tf.function
def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

@tf.function
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# aliases
mse = MSE = mean_squared_error

@tf.function
def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))

@tf.function
def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5
@tf.function
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

mae = MAE = mean_absolute_error