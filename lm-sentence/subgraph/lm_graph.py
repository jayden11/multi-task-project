from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import pdb


def lm_private(encoder_units, pos_prediction, chunk_prediction,config, is_training ,
                sentence_lengths):
    """Decode model for lm

    Args:
        encoder_units - these are the encoder units:
        [config.batch_size X encoder_size] with the one the pos prediction
        pos_prediction:
        must be the same size as the encoder_size

    returns:
        logits
    """
    # concatenate the encoder_units and the pos_prediction

    pos_prediction = tf.reshape(pos_prediction,
        [config.batch_size, config.num_steps, config.pos_embedding_size])
    chunk_prediction = tf.reshape(chunk_prediction,
        [config.batch_size, config.num_steps, config.chunk_embedding_size])
    lm_inputs = tf.concat(2, [chunk_prediction, pos_prediction, encoder_units])

    with tf.variable_scope("lm_decoder"):
        if config.bidirectional == True:
            if config.lstm == True:
                cell_fw = rnn_cell.BasicLSTMCell(config.lm_decoder_size, forget_bias=1.0)
                cell_bw = rnn_cell.BasicLSTMCell(config.lm_decoder_size, forget_bias=1.0)
            else:
                cell_fw = rnn_cell.GRUCell(config.lm_decoder_size)
                cell_bw = rnn_cell.GRUCell(config.lm_decoder_size)

            if is_training and config.keep_prob < 1:
                cell_fw = rnn_cell.DropoutWrapper(
                    cell_fw, output_keep_prob=config.keep_prob)
                cell_bw = rnn_cell.DropoutWrapper(
                    cell_bw, output_keep_prob=config.keep_prob)

            cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
            cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

            initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
            initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

            # this function puts the 3d tensor into a 2d tensor: config.batch_size x input size
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, config.num_steps,
                                             lm_inputs)]

            decoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw,
                                                      inputs, initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      sequence_length=sentence_lengths,
                                                      scope="lm_rnn")
            output = tf.reshape(tf.concat(1, decoder_outputs),
                                [-1, 2*config.lm_decoder_size])
            softmax_w = tf.get_variable("softmax_w",
                                        [2*config.lm_decoder_size,
                                         config.vocab_size])
        else:
            if config.lstm == True:
                cell = rnn_cell.BasicLSTMCell(config.lm_decoder_size)
            else:
                cell = rnn_cell.GRUCell(config.lm_decoder_size)

            if is_training and config.keep_prob < 1:
                cell = rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)

            cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

            initial_state = cell.zero_state(config.batch_size, tf.float32)

            # this function puts the 3d tensor into a 2d tensor: config.batch_size x input size
            inputs = [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, config.num_steps,
                                             lm_inputs)]

            decoder_outputs, decoder_states = rnn.rnn(cell,
                                                      inputs, initial_state=initial_state,
                                                      sequence_length=sentence_lengths,
                                                      scope="lm_rnn")

            output = tf.reshape(tf.concat(1, decoder_outputs),
                                [-1, config.lm_decoder_size])
            softmax_w = tf.get_variable("softmax_w",
                                        [config.lm_decoder_size,
                                         config.vocab_size])

        softmax_b = tf.get_variable("softmax_b", [config.vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        l2_penalty = tf.reduce_sum(tf.square(output))

    return logits, l2_penalty
