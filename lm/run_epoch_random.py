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
from graph import Shared_Model

import saveload


def run_epoch(session, m, conll_words, ptb_words, pos, ptb_pos, chunk, ptb_chunk, pos_vocab_size,
            chunk_vocab_size, vocab_size, num_steps, verbose=False, valid=False, model_type='JOINT'):
    """Runs the model on the given data."""
    # =====================================
    # Initialise variables
    # =====================================
    conll_epoch_size = (len(conll_words) // m.batch_size) + 1
    ptb_epoch_size = (len(ptb_words) // m.batch_size) + 1
    epoch_stats = {
        'comb_loss': 0.0,
        "pos_total_loss": 0.0,
        "chunk_total_loss": 0.0,
        "lm_total_loss": 0.0,
        "iters": 0,
        "accuracy": 0.0,
        "pos_predictions": [],
        "pos_true": [],
        "chunk_predictions": [],
        "chunk_true": [],
        "lm_predictions": [],
        "lm_true": []
    }


    print('creating batches')

    conll_batches = reader.create_batches(conll_words, pos, chunk, m.batch_size,
                            m.num_steps, pos_vocab_size, chunk_vocab_size, vocab_size, continuing=True)

    conll_iter = 0

    ptb_batches = reader.create_batches(ptb_words, ptb_pos, ptb_chunk, m.batch_size,
                            m.num_steps, pos_vocab_size, chunk_vocab_size, vocab_size, continuing=True)
    ptb_iter = 0

    # =======================================================
    # Define the train batch method
    # -------------------------------------------------------
    # We're then going to use this method to train a batch of
    # each data type in turn
    # ======================================================

    def train_batch(batch, eval_op, model_type, epoch_stats, stop_write=False):
        (x, y_pos, y_chunk, y_lm, sentence_lengths) = batch

        joint_loss, _, pos_int_pred, chunk_int_pred, lm_int_pred, pos_int_true, \
            chunk_int_true, lm_int_true, pos_loss, chunk_loss, lm_loss = \
            session.run([m.joint_loss, eval_op, m.pos_int_pred,
                         m.chunk_int_pred, m.lm_int_pred, m.pos_int_targ, m.chunk_int_targ,
                         m.lm_int_targ, m.pos_loss, m.chunk_loss, m.lm_loss],
                        {m.input_data: x,
                         m.pos_targets: y_pos,
                         m.chunk_targets: y_chunk,
                         m.lm_targets: y_lm,
                         m.sentence_lengths: sentence_lengths})

        epoch_stats["comb_loss"] += joint_loss
        epoch_stats["chunk_total_loss"] += chunk_loss
        epoch_stats["pos_total_loss"] += pos_loss
        epoch_stats["lm_total_loss"] += lm_loss
        epoch_stats["iters"] += 1

        if verbose and (epoch_stats["iters"] % 10 == 0):
            if model_type == 'POS':
                costs = epoch_stats["pos_total_loss"]
                cost = pos_loss
            elif model_type == 'CHUNK':
                costs = epoch_stats["chunk_total_loss"]
                cost = chunk_loss
            elif model_type == 'LM':
                costs = epoch_stats["lm_total_loss"]
                cost = lm_loss
            else:
                costs = epoch_stats["comb_loss"]
                cost = joint_loss
            print("Type: %s,cost: %3f, step: %3f" % (model_type, cost, epoch_stats['iters']))

        if model_type != "LM" and stop_write==False:
            pos_int_pred = np.reshape(pos_int_pred, [m.batch_size, m.num_steps])
            pos_int_true = np.reshape(pos_int_true, [m.batch_size, m.num_steps])
            epoch_stats["pos_predictions"].append(pos_int_pred)
            epoch_stats["pos_true"].append(pos_int_true)

            chunk_int_pred = np.reshape(chunk_int_pred, [m.batch_size, m.num_steps])
            chunk_int_true = np.reshape(chunk_int_true, [m.batch_size, m.num_steps])
            epoch_stats["chunk_predictions"].append(chunk_int_pred)
            epoch_stats["chunk_true"].append(chunk_int_true)

            lm_int_pred = np.reshape(lm_int_pred, [m.batch_size, m.num_steps])
            lm_int_true = np.reshape(lm_int_true, [m.batch_size, m.num_steps])
            epoch_stats["lm_predictions"].append(lm_int_pred)
            epoch_stats["lm_true"].append(lm_int_true)

        return epoch_stats

    # ==========================================================
    # Do the epoch
    # ----------------------------------------------------------
    # randomly choose a dataset, and then increment your counter
    # ==========================================================

    if valid:
        eval_op = tf.no_op()
        for i in range(conll_epoch_size):
            train_batch(conll_batches, i, eval_op, "JOINT", epoch_stats)
    else:
        print('ptb epoch size: ' + str(ptb_epoch_size))
        print('conll epoch size: ' + str(conll_epoch_size))
        while (ptb_iter < ptb_epoch_size) or (conll_iter < conll_epoch_size):
            if np.random.rand(1) < m.mix_percent:
                eval_op = m.joint_op
                epoch_stats = train_batch(next(conll_batches), \
                    eval_op, "JOINT", epoch_stats, (conll_iter > conll_epoch_size))
                conll_iter +=1
                # print('conll iter: ' + str(conll_iter))
            else:
                eval_op = m.auto_op
                epoch_stats = train_batch(next(ptb_batches), \
                    eval_op, "LM", epoch_stats)
                ptb_iter += 1
                # print('ptb iter: ' + str(ptb_iter))
        eval_op = m.joint_op
        epoch_stats = train_batch(next(conll_batches), \
            eval_op, "JOINT", epoch_stats, (conll_iter > conll_epoch_size))
        conll_iter +=1

    return (epoch_stats["comb_loss"] / epoch_stats["iters"]), \
        epoch_stats["pos_predictions"], epoch_stats["chunk_predictions"], \
        epoch_stats["lm_predictions"], epoch_stats["pos_true"], \
        epoch_stats["chunk_true"], epoch_stats["lm_true"], \
        (epoch_stats["pos_total_loss"] / epoch_stats["iters"]), \
        (epoch_stats["chunk_total_loss"] / epoch_stats["iters"]), \
        (epoch_stats["lm_total_loss"] / epoch_stats["iters"])
