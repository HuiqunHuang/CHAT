# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Reshape
from tensorflow.python.layers.base import Layer

class BiLSTMModule(Layer):
    def __init__(self, l, num_hidden, num_classes):
        super(BiLSTMModule, self).__init__(name="BiLSTMModule")
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.timesteps = l


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_hidden': self.num_hidden,
            'num_classes': self.num_classes,
            'weight': self.w,
            'biase': self.b
        })
        return config

    def aa(self, bb):
        data = bb
        return data

    def BiRNN(self, x):
        forward = ForwardBackwardlstm(self.num_hidden)
        backward = ForwardBackwardlstm(self.num_hidden)
        h_s = []
        for i in range(0, self.timesteps):
            xt = x[:, i, :]
            xt = Reshape((1, self.num_classes))(xt)
            f_h_s, f_c_s = forward.lstmoperation(xt)

            xt = x[:, self.timesteps-i-1, :]
            xt = Reshape((1, self.num_classes))(xt)
            b_h_s, b_c_s = backward.lstmoperation(xt)

            a = tf.concat([f_h_s, b_h_s], axis=-1)
            h_s.append(a)
        h_s = tf.stack(h_s, axis=1)

        return h_s

class ForwardBackwardlstm(Layer):
    def __init__(self, p):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(ForwardBackwardlstm, self).__init__(name="ForwardBackwardlstm")
        self.lstm = LSTM(p, return_state=True)
        self.initial_state = None

    def lstmoperation(self, x):
        """
        x : input data (shape = batch,1,n)
        """
        h_s, _, c_s = self.lstm(x, initial_state=self.initial_state)
        self.initial_state = [h_s, c_s]
        return h_s, c_s

    def reset_state(self, h0, c0):
        self.initial_state = [h0, c0]

