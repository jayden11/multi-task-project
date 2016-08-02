from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import pdb


def _chunk_private(encoder_units, pos_prediction, pos_hidden, config):
    """Decode model for chunks

    Args:
        encoder_units - these are the encoder units:
        [batch_size X encoder_size] with the one the pos prediction
        pos_prediction:
        must be the same size as the encoder_size

    returns:
        logits
    """
    # concatenate the encoder_units and the pos_prediction

    pos_prediction = tf.reshape(pos_prediction,
        [batch_size, num_steps, pos_embedding_size])
    #pos_hidden = tf.reshape(pos_hidden, [batch_size, num_steps, 2*config.pos_decoder_size])
    chunk_inputs = tf.concat(2, [pos_prediction, encoder_units])

    with tf.variable_scope("chunk_decoder"):
        if config.bidirectional == True:
            if config.lstm == True:
                cell_fw = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)
                cell_bw = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)
            else:
                cell_fw = rnn_cell.GRUCell(config.chunk_decoder_size)
                cell_bw = rnn_cell.GRUCell(config.chunk_decoder_size)

            if is_training and config.keep_prob < 1:
                cell_fw = rnn_cell.DropoutWrapper(
                    cell_fw, output_keep_prob=config.keep_prob)
                cell_bw = rnn_cell.DropoutWrapper(
                    cell_bw, output_keep_prob=config.keep_prob)

            cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
            cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

            initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

            # this function puts the 3d tensor into a 2d tensor: batch_size x input size
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, num_steps,
                                             chunk_inputs)]

            decoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw,
                                                      inputs, initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      sequence_length=sentence_lengths,
                                                      scope="chunk_rnn")
            output = tf.reshape(tf.concat(1, decoder_outputs),
                                [-1, 2*config.chunk_decoder_size])
            softmax_w = tf.get_variable("softmax_w",
                                        [2*config.chunk_decoder_size,
                                         num_chunk_tags])
        else:
            if config.lstm == True:
                cell = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)
            else:
                cell = rnn_cell.GRUCell(config.chunk_decoder_size)

            if is_training and config.keep_prob < 1:
                cell = rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)

            cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

            initial_state = cell.zero_state(config.batch_size, tf.float32)

            # this function puts the 3d tensor into a 2d tensor: batch_size x input size
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, num_steps,
                                             chunk_inputs)]

            decoder_outputs, decoder_states = rnn.rnn(cell,
                                                      inputs, initial_state=initial_state,
                                                      sequence_length=sentence_lengths,
                                                      scope="chunk_rnn")

            output = tf.reshape(tf.concat(1, decoder_outputs),
                                [-1, config.chunk_decoder_size])

            softmax_w = tf.get_variable("softmax_w",
                                        [config.chunk_decoder_size,
                                         num_chunk_tags])

        softmax_b = tf.get_variable("softmax_b", [num_chunk_tags])
        logits = tf.matmul(output, softmax_w) + softmax_b
        l2_penalty = tf.reduce_sum(tf.square(output))

    return logits, output, l2_penalty
