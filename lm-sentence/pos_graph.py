from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import pdb


def pos_private(encoder_units, config):
    """Decode model for pos

    Args:
        encoder_units - these are the encoder units
        num_pos - the number of pos tags there are (output units)

    returns:
        logits
    """
    with tf.variable_scope("pos_decoder"):
        if config.bidirectional == True:
            if config.lstm == True:
                cell_fw = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                              forget_bias=1.0)
                cell_bw = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                              forget_bias=1.0)
            else:
                cell_fw = rnn_cell.GRUCell(config.pos_decoder_size)
                cell_bw = rnn_cell.GRUCell(config.pos_decoder_size)

            if is_training and config.keep_prob < 1:
                cell_fw = rnn_cell.DropoutWrapper(
                    cell_fw, output_keep_prob=config.keep_prob)
                cell_bw = rnn_cell.DropoutWrapper(
                    cell_bw, output_keep_prob=config.keep_prob)

            cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
            cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

            initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

            # puts it into batch_size X input_size
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, num_steps,
                                             encoder_units)]

            decoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      sequence_length=sentence_lengths,
                                                      scope="pos_rnn")

            output = tf.reshape(tf.concat(1, decoder_outputs),
                                [-1, 2*config.pos_decoder_size])

            softmax_w = tf.get_variable("softmax_w",
                                        [2*config.pos_decoder_size,
                                         num_pos_tags])
        else:
            if config.lstm == True:
                cell = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                          forget_bias=1.0)
            else:
                cell = rnn_cell.GRUCell(config.pos_decoder_size)

            if is_training and config.keep_prob < 1:
                cell = rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)

            cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

            initial_state = cell.zero_state(config.batch_size, tf.float32)

            # puts it into batch_size X input_size
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, num_steps,
                                             encoder_units)]

            decoder_outputs, decoder_states = rnn.rnn(cell, inputs,
                                                      initial_state=initial_state,
                                                      scope="pos_rnn")
            output = tf.reshape(tf.concat(1, decoder_outputs),
                                [-1, config.pos_decoder_size])

            softmax_w = tf.get_variable("softmax_w",
                                        [config.pos_decoder_size,
                                         num_pos_tags])

        softmax_b = tf.get_variable("softmax_b", [num_pos_tags])
        logits = tf.matmul(output, softmax_w) + softmax_b
        l2_penalty = tf.reduce_sum(tf.square(output))

    return logits, output, l2_penalty
