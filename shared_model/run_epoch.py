from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import random

import tensorflow as tf
import tensorflow.python.platform

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import model_reader as reader
import numpy as np
import pdb
import pandas as pd
from graph import Shared_Model


def run_epoch(session, m, words, pos, chunk, pos_vocab_size, chunk_vocab_size,
              verbose=False, valid=False):
    """Runs the model on the given data."""
    epoch_size = ((len(words) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    pos_predictions = []
    pos_true = []
    chunk_predictions = []
    chunk_true = []
    state = m.initial_state.eval()

    for step, (x, y_pos, y_chunk) in enumerate(reader.create_batches(words, pos, chunk, m.batch_size,
                                               m.num_steps, pos_vocab_size, chunk_vocab_size)):
        # probabilistic task choice
        pos_or_chunk = random.choice([True, False])
        if pos_or_chunk:
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.pos_op # m.joint_op
            loss = m.pos_loss
            final_state = m.pos_last_state
        else:
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.pos_op # m.chunk_op # m.joint_op
            loss = m.chunk_loss
            final_state = m.chunk_last_state


        cost, state, _, pos_int_pred, chunk_int_pred, pos_int_true, chunk_int_true = \
            session.run([loss, final_state, eval_op, m.pos_int_pred,
                         m.chunk_int_pred, m.pos_int_targ, m.chunk_int_targ],
                        {m.input_data: x,
                         m.pos_targets: y_pos,
                         m.chunk_targets: y_chunk,
                         m.initial_state: state})
        costs += cost
        iters += 1
        if verbose and step % (epoch_size // 10) == 0:
            type_st = "Pos" if pos_or_chunk else "Chunk"
            print("Type: %s,cost: %3f, total cost: %3f" % (type_st, cost, costs))

        pos_int_pred = np.reshape(pos_int_pred, [m.batch_size, m.num_steps])
        pos_predictions.append(pos_int_pred)
        pos_true.append(pos_int_true)

        chunk_int_pred = np.reshape(chunk_int_pred, [m.batch_size, m.num_steps])
        chunk_predictions.append(chunk_int_pred)
        chunk_true.append(chunk_int_true)

    return (costs / iters), pos_predictions, chunk_predictions, pos_true, chunk_true
