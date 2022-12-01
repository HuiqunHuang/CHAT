from __future__ import print_function
import tensorflow as tf

from Model.Layers.BiLSTM import BiLSTMModule
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import random_normal
from tensorflow.python.keras.layers import BatchNormalization, Activation, Reshape, Dense, Permute
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.base import Layer


def categoricalTimeSeriesDataPredictionModel(c_conf, batch_size, out_height, out_width, category_num):
    main_inputs = []
    len_seq, feature_num, map_height, map_width = c_conf
    L = feature_num * len_seq
    num_hidden = int(map_height * map_width / 4)
    num_classes = map_height * map_width
    input = Input(shape=(feature_num * len_seq, map_height, map_width))
    main_inputs.append(input)
    input = Reshape((L, out_height * out_width))(input)
    print("Shape of input:")
    print(input.shape, batch_size)
    print("main_inputs:")
    print(main_inputs)

    bilstm = BiLSTMModule(L, num_hidden, num_classes)
    bilstm_h_s = bilstm.BiRNN(input)
    bilstm_h_s = BatchNormalization()(bilstm_h_s)
    print("bilstm_h_s: " + str(bilstm_h_s))
    twa = TemporalWiseAttentionModule(L, num_hidden*2, num_classes)
    twa_h_s = twa.attentionoperation(bilstm_h_s)
    twa_h_s = BatchNormalization()(twa_h_s)
    print("twa_h_s: " + str(twa_h_s))


    EI = input
    print("EI:")
    print(EI)
    EJ = Permute((2, 1))(input)
    print("EJ:")
    print(EJ)

    iam = InteractionAttentionModule(L, num_hidden, num_classes)
    main_output = iam.attentionoperation(EI, EJ, bilstm_h_s, batch_size)

    main_output = Dense(category_num)(main_output)
    main_output = Activation('relu')(main_output)
    print(main_inputs)
    print(main_output)

    model = Model(main_inputs, main_output)

    return model


class TemporalWiseAttentionModule(Layer):
    def __init__(self, l, num_hidden, num_classes):
        super(TemporalWiseAttentionModule, self).__init__(name="TemporalWiseAttentionModule")
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.timesteps = l
        self.w = Dense(num_hidden)
        self.b = Dense(l)

    def aa(self, bb):
        data = bb
        return data

    def attentionoperation(self, h_s):
        score = tf.nn.tanh(self.w(h_s))
        attention_weights = tf.nn.softmax(score)
        h_s = tf.multiply(h_s, attention_weights)

        return h_s


class InteractionAttentionModule(tf.keras.layers.Layer):
    def __init__(self, l, num_hidden, num_classes):
        super(InteractionAttentionModule, self).__init__(name="InteractionAttentionModule")
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.timesteps = l
        self.wij = Dense(num_classes)
        self.wik = Dense(1)
        self.wjk = Dense(num_classes)
        self.wint = Dense(num_classes)
        self.u = tf.Variable(initial_value=random_normal(shape=[1, num_classes], mean=1, stddev=1.0, dtype=tf.float32,
                                                         seed=1), name="u", trainable=True)
        self.v = tf.Variable(initial_value=random_normal(shape=[num_hidden, 1], mean=1, stddev=1.0, dtype=tf.float32,
                                                         seed=1), name="v", trainable=True)


    def build(self, input_shape):
        self.built = True

        self.u = self.add_weight("u", [1, self.num_classes], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                 trainable=True)
        self.v = self.add_weight("v", [self.num_hidden, 1], dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                 trainable=True)

    def attentionoperation(self, EI, EJ, hiddenstates, batch_size):
        # EI: region
        # EJ: anomaly events
        # EK: time slot
        # X: hidden states
        EI = tf.unstack(EI, self.timesteps, 1)
        print('hiddenstates', hiddenstates.shape)
        hiddenstates = tf.unstack(hiddenstates, self.timesteps, 1)
        return hiddenstates[-1]


if __name__ == '__main__':
    print("")

