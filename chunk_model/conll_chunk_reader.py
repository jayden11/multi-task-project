
"""Utilities for parsing CONll text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import defaultdict
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
    (2). Read in the chunkss, add padding, and assign each a category of the form
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
    # get rid of all words with frequency == 1
    counter = {k: v for k, v in counter.items() if v > 1}
    counter['<unk>'] = 10000
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    # add in unknown token at the beginning, so with index 1
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


"""
    1.1. Word Methods
"""


def _file_to_word_ids(filename, word_to_id, padding_width):
    # assumes _build_vocab has been called first as is called word to id
    data = _read_tokens(filename, padding_width, 0)
    default_value = word_to_id['<unk>']
    return [word_to_id.get(word, default_value) for word in data]

"""
    1.2. chunk Methods
"""


def _int_to_chunk(chunk_int, chunk_vocab_size):
    a = np.empty(chunk_vocab_size)
    a.fill(0)
    np.put(a, chunk_int-1, 1)
    return a


def _seq_chunk(chunk_integers, chunk_vocab_size):
    return np.vstack(_int_to_chunk(
                     chunk, chunk_vocab_size) for chunk in chunk_integers)


def _file_to_chunk_classifications(filename, chunk_to_id, padding_width):
    # assumes _build_vocab has been called first and is called chunk to id
    data = _read_tokens(filename, padding_width, 2)
    default_value = chunk_to_id['<unk>']
    return [chunk_to_id.get(tag, default_value) for tag in data]


def _raw_x_y_data(data_path, num_steps):
    train = "train_custom.txt"
    valid = "val_custom.txt"

    train_path = os.path.join(data_path, train)
    valid_path = os.path.join(data_path, valid)

    word_to_id = _build_vocab(train_path, num_steps-1, 0)
    chunk_to_id = _build_vocab(train_path, num_steps-1, 2)

    word_data_t = _file_to_word_ids(train_path, word_to_id, num_steps-1)
    chunk_data_t = _file_to_chunk_classifications(train_path, chunk_to_id, num_steps-1)

    word_data_v = _file_to_word_ids(valid_path, word_to_id, num_steps-1)
    chunk_data_v = _file_to_chunk_classifications(valid_path, chunk_to_id, num_steps-1)

    word_vocab = len(word_to_id)
    chunk_vocab = len(chunk_to_id)
    return word_data_t, chunk_data_t, word_data_v, chunk_data_v, word_vocab, chunk_vocab, word_to_id, chunk_to_id


"""
============
Section 2.
============

Here we want to feed in the raw data, batch-size, and window size
 and get back mini batches. These will be of size [batch_size, num_steps]

Args:
raw_words = the raw array of the word integers
raw_chunk = the raw array of the chunk integers
batch_size = batch size
num_steps = the number of steps you are going to look back in your rnn
chunk_vocab_size = the size of the the number of chunk tokens (
    needed for transfer into the [0,...,1,...,0] format)

Yields
(x,y) - x the batch, y the chunk tags

"""


def _create_batches(raw_words, raw_chunk, batch_size, num_steps, chunk_vocab_size):
    raw_words = np.array(raw_words, dtype=np.int32)
    raw_chunk = np.array(raw_chunk, dtype=np.int32)

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
    2. Prepare the output (chunk) data
    """

    chunk_data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        chunk_data[i] = raw_chunk[batch_len * i:batch_len * (i + 1)]

    """
    3. Do the epoch thing and iterate
    """

    # how many times do you iterate to reach the end of the epoch
    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = word_data[:, i*num_steps:(i+1)*num_steps]
        y = np.vstack(_seq_chunk(chunk_data[chunk, i*num_steps:(i+1)*num_steps],
                      chunk_vocab_size) for chunk in range(batch_size))
        y = y.astype(np.int32)
        yield (x, y)

def int_to_string(int_pred, d):
    # integers are the Values
    keys = []
    for x in int_pred:
        keys.append([k for k, v in d.iteritems() if v == x])

    return keys
