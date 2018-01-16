"""
    TODO:
        Flexible hparams setting
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
from tflibs import *
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

from configs import hparams

def nipsHead(x):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 84, 84, 4]; output: [None, 2592] -> [None, 256];
    '''
    print('Using nips head design')
    x = tf.nn.relu(conv2d(x, 32, "l1", [3, 3], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 32, "l2", [3, 3], [2, 2], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, hparams.cell_size, "fc", normalized_columns_initializer(0.01)))
    return x

def deepHead(x):
    """ My experiments usages head
        input: [None, 32, 32, 5]; output [None, 576] --> [None, 512] -> [None, 256]
        TODO: batchnorm is needed
    """
    print ("Using our experimental head design")
    x = tf.nn.relu(conv2d(x, 64, "l1", [3, 3], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l2", [6, 6], [4, 4], pad="VALID"))
    # pooling ?
    x = flatten(x)
    # add one more layer
    x = tf.nn.relu(linear(x, 512, "fc1", normalized_columns_initializer(0.01)))
    x = tf.nn.relu(linear(x, hparams.cell_size, "fc2", normalized_columns_initializer(0.01)))
    return x

class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))

        x = nipsHead(x)
        x = tf.expand_dims(x, [0])

        size = hparams.cell_size
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [obs], self.state_in[0]: c, self.state_in[1]: h})

    def act_inference(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [obs], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, obs, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [obs], self.state_in[0]: c, self.state_in[1]: h})[0]
