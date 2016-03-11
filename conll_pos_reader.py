
"""Utilities for parsing CONll text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
import pandas as pd
import pdb

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

"""
==========
Section 1.
==========

This section defines the methods that:
    (1). Read in the words, add padding, and assign each an integer
    (2). Read in the POSs, add padding, and assign each a category of the form
    [0,...,1,...0] etc.

Section 2 will deal with creating the window and mini-batching

"""

"""
    1.0. Utility Methods
"""


def _read_tokens(filename, padding_val, col_val):
    # Col Values
    # 0 - words
    # 1 - POS
    # 2 - Chunks

    words = pd.read_csv(filename, sep=' ', header=None)[col_val].as_matrix()
    # padding token '0'
    return np.pad(
        words, pad_width=(padding_val, 0), mode='constant', constant_values=0)


def _build_vocab(filename, padding_width, col_val):
    # can be used for classifications and input vocab
    data = _read_tokens(filename, padding_width, col_val)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


"""
    1.1. Word Methods
"""


def _file_to_word_ids(filename, word_to_id, padding_width):
    # assumes _build_vocab has been called first as is called word to id
    data = _read_tokens(filename, padding_width, 0)
    return [word_to_id[word] for word in data]

"""
    1.2. POS Methods
"""


def _int_to_POS(pos_int, POS_vocab_size):
    a = np.empty(POS_vocab_size)
    a.fill(0)
    np.put(a, pos_int-1, 1)
    return a


def _file_to_POS_classifications(filename, POS_to_id, padding_width):
    # assumes _build_vocab has been called first and is called POS to id
    data = _read_tokens(filename, padding_width, 1)
    return [POS_to_id[word] for word in data]


def _raw_x_y_data(data_path, num_steps):
    train_path = os.path.join(data_path, "train_custom.txt")
    valid_path = os.path.join(data_path, "val_custom.txt")
    
    word_to_id = _build_vocab(train_path, num_steps-1, 0)
    POS_to_id = _build_vocab(train_path, num_steps-1, 1)

    word_data = _file_to_word_ids(train_path, word_to_id, num_steps-1)
    pos_data = _file_to_POS_classifications(train_path, POS_to_id, num_steps-1)

    word_vocab_size = len(word_to_id)
    pos_vocab_size = len(POS_to_id)
    return word_data, pos_data, word_vocab_size, pos_vocab_size


"""
============
Section 2.
============

Here we want to feed in the raw data, batch-size, and window size
 and get back mini batches. These will be of size [batch_size, num_steps]

Args:
raw_words = the raw array of the word integers
raw_pos = the raw array of the pos integers
batch_size = batch size
num_steps = the number of steps you are going to look back in your rnn
pos_vocab_size = the size of the the number of pos tokens (needed for transfer
    into the [0,...,1,...,0] format)

Yields
(x,y) - x the batch, y the pos tags

"""


def _create_batches(raw_words, raw_pos, batch_size, num_steps, pos_vocab_size):
    raw_words = np.array(raw_words, dtype=np.int32)
    raw_pos = np.array(raw_pos, dtype=np.int32)

    """
    1. Prepare the input (word) data
    """

    data_len = len(raw_words)
    # We're going to reshape the data into [batch_size, batch_length]
    # and then slice up the batch length to create our batches
    batch_len = data_len // batch_size
    word_data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        word_data[i] = raw_words[batch_len * i:batch_len * (i + 1)]

    """
    2. Prepare the output (POS) data
    """

    pos_data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        pos_data[i] = raw_pos[batch_len * i:batch_len * (i + 1)]

    """
    3. Do the epoch thing and iterate
    """

    # how many times do you iterate to reach the end of the epoch
    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = word_data[:, i*num_steps:(i+1)*num_steps]
        y = np.vstack(_int_to_POS(pos, pos_vocab_size) for pos in pos_data[
            :, (i+1)*num_steps])
        y = y.astype(np.int32)
        yield (x, y)
