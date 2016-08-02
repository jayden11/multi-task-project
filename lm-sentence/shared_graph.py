from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import pdb


def shared_layer(input_data, config):
    """Build the model to decoding

    Args:
        input_data = size batch_size X num_steps X embedding size

    Returns:
        output units
    """

    if config.bidirectional == True:
        if config.lstm == True:
            cell_fw = rnn_cell.BasicLSTMCell(config.encoder_size, forget_bias = 1.0)
            cell_bw = rnn_cell.BasicLSTMCell(config.encoder_size, forget_bias = 1.0)
        else:
            cell_fw = rnn_cell.GRUCell(config.encoder_size)
            cell_bw = rnn_cell.GRUCell(config.encoder_size)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, input_data)]

        if is_training and config.keep_prob < 1:
            cell_fw = rnn_cell.DropoutWrapper(
                cell_fw, output_keep_prob=config.keep_prob)
            cell_bw = rnn_cell.DropoutWrapper(
                cell_bw, output_keep_prob=config.keep_prob)


        cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
        cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

        initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

        encoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw,
                                              sequence_length=sentence_lengths,
                                              scope="encoder_rnn")


    else:
        if config.lstm == True:
            cell = rnn_cell.BasicLSTMCell(config.encoder_size)
        else:
            cell = rnn_cell.GRUCell(config.encoder_size)

        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, input_data)]

        if is_training and config.keep_prob < 1:
            cell = rnn_cell.DropoutWrapper(
                cell, output_keep_prob=config.keep_prob)

        cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

        initial_state = cell.zero_state(config.batch_size, tf.float32)

        encoder_outputs, encoder_states = rnn.rnn(cell, inputs,
                                                  initial_state=initial_state,
                                                  sequence_length=sentence_lengths,
                                                  scope="encoder_rnn")

    return encoder_outputs
