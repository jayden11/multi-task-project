"""Utilities for parsing CONll text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
#import pandas as pd
import csv
import pdb
import pickle

import numpy as np
#from six.moves import xrange  # pylint: disable=redefined-builtin

"""
==========
Section 1.
==========
This section defines the methods that:
    (1). Read in the words, add padding, and assign each an integer
    (2). Read in the tagss, add padding, and assign each a category of the form
    [0,...,1,...0] etc.
Section 2 will deal with creating the window and mini-batching
"""

"""
    1.0. Utility Methods
"""


def read_tokens(filename, padding_val, col_val=-1):
    # Col Values
    # 0 - words
    # 1 - POS
    # 2 - tags
    # -1 - for everything

    with open(filename, 'rt', encoding='utf8') as csvfile:
            r = csv.reader(csvfile, delimiter=' ')
            words = np.transpose(np.array([x for x in list(r) if x != []])).astype(object)
    # padding token '0'
    print('reading' + filename)
    if col_val!=-1:
        words = words[col_val]
    words = np.pad(
        words, pad_width=(padding_val, 0), mode='constant', constant_values=0)
    if col_val!=-1:
        return [str(x).lower() for x in words]
    else:
        return words

def import_embeddings(filename):
    words = {}
    with open(filename, 'rt', encoding='utf8') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        for row in r:
            words[row[0]] = row[1:]
    return words

def _build_vocab(filename, ptb_filename, padding_width, col_val):
    # can be used for input vocab
    conll_data = read_tokens(filename, padding_width, col_val)
    ptb_data = read_tokens(ptb_filename, padding_width, col_val)
    data = np.concatenate((conll_data, ptb_data))
    counter = collections.Counter(data)
    # get rid of all words with frequency == 1
    counter = {k: v for k, v in counter.items() if (v > 1)}
    counter['<unk>'] = 10000
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _build_vocab_embedding(filename, ptb_filename, padding_width, col_val, embedding):
    # can be used for input vocab
    conll_data = read_tokens(filename, padding_width, col_val)
    ptb_data = read_tokens(ptb_filename, padding_width, col_val)
    data = np.concatenate((conll_data, ptb_data))
    counter = collections.Counter(data)
    # get rid of all words with frequency == 1
    counter = {k: v for k, v in counter.items() if (v > 1) or (k not in embedding)}
    counter['<unk>'] = 10000
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _build_tags(filename, padding_width, col_val):
    # can be used for classifications and input vocab
    data = read_tokens(filename, padding_width, col_val)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    tag_to_id = dict(zip(words, range(len(words))))

    return tag_to_id


"""
    1.1. Word Methods
"""


def _file_to_word_ids(filename, word_to_id, padding_width):
    # assumes _build_vocab has been called first as is called word to id
    data = read_tokens(filename, padding_width, 0)
    default_value = word_to_id['<unk>']
    return [word_to_id.get(word, default_value) for word in data]

"""
    1.2. tag Methods
"""


def _int_to_tag(tag_int, tag_vocab_size):
    # creates the one-hot vector
    a = np.empty(tag_vocab_size)
    a.fill(0)
    np.put(a, tag_int, 1)
    return a


def _seq_tag(tag_integers, tag_vocab_size):
    # create the array of one-hot vectors for your sequence
    return np.vstack(_int_to_tag(
                     tag, tag_vocab_size) for tag in tag_integers)


def _file_to_tag_classifications(filename, tag_to_id, padding_width, col_val):
    # assumes _build_vocab has been called first and is called tag to id
    data = read_tokens(filename, padding_width, col_val)
    return [tag_to_id[tag] for tag in data]


def raw_x_y_data(data_path, num_steps, ptb_data_path, embedding=False, embedding_path=None):
    train = "train.txt"
    valid = "validation.txt"
    train_valid = "train_val_combined.txt"
    comb = "all_combined.txt"
    test = "test.txt"
    ptb = 'train.txt'

    train_path = os.path.join(data_path, train)
    valid_path = os.path.join(data_path, valid)
    train_valid_path = os.path.join(data_path, train_valid)
    comb_path = os.path.join(data_path, comb)
    test_path = os.path.join(data_path, test)
    ptb_path = os.path.join(ptb_data_path, ptb)

    if os.path.exists(comb_path) != True:
        print('writing combined')
        test_data = pd.read_csv(test_path, sep= ' ',header=None)
        train_data = pd.read_csv(train_path, sep= ' ',header=None)
        valid_data = pd.read_csv(valid_path, sep= ' ', header=None)

        comb = pd.concat([train_data,valid_data,test_data])
        comb.to_csv(os.path.join(data_path,'critic_all_combined.txt'), sep=' ', index=False, header=False)

    if os.path.exists(train_valid_path) != True:
        print('writing combined')
        valid_data = pd.read_csv(valid_path, sep= ' ',header=None)
        train_data = pd.read_csv(train_path, sep= ' ',header=None)

        comb = pd.concat([train_data,valid_data])
        comb.to_csv(os.path.join(data_path,'critic_train_val_combined.txt'), sep=' ', index=False, header=False)


    if embedding == True:
        word_embedding_full = import_embeddings(embedding_path)
        word_to_id = _build_vocab_embedding(comb_path, ptb_path, num_steps-1, 0, word_embedding_full)
        id_to_word = {v: k for k, v in word_to_id.items()}
        ordered_vocab = [id_to_word[i] for i in range(len(id_to_word))]
        embedding_len = len(word_embedding_full['the'])
        word_embedding = [word_embedding_full.get(key.lower(), np.random.randn(embedding_len))
                                   for key in ordered_vocab]
    else:
        word_to_id = _build_vocab(comb_path, ptb_path, num_steps-1, 0)
        word_embedding = None

    # use the full training set for building the target tags
    pos_to_id = _build_tags(comb_path, num_steps-1, 1)
    chunk_to_id = _build_tags(comb_path, num_steps-1, 2)

    word_data_t = _file_to_word_ids(train_path, word_to_id, num_steps-1)
    pos_data_t = _file_to_tag_classifications(train_path, pos_to_id, num_steps-1, 1,)
    chunk_data_t = _file_to_tag_classifications(train_path, chunk_to_id, num_steps-1, 2)

    ptb_word_data = _file_to_word_ids(ptb_path, word_to_id, num_steps-1)
    ptb_pos_data = _file_to_tag_classifications(ptb_path, pos_to_id, num_steps-1, 1)
    ptb_chunk_data = _file_to_tag_classifications(ptb_path, chunk_to_id, num_steps-1, 2)

    word_data_v = _file_to_word_ids(valid_path, word_to_id, num_steps-1)
    pos_data_v = _file_to_tag_classifications(valid_path, pos_to_id, num_steps-1, 1)
    chunk_data_v = _file_to_tag_classifications(valid_path, chunk_to_id, num_steps-1, 2)

    word_data_c = _file_to_word_ids(train_valid_path, word_to_id, num_steps-1)
    pos_data_c = _file_to_tag_classifications(train_valid_path, pos_to_id, num_steps-1, 1)
    chunk_data_c = _file_to_tag_classifications(train_valid_path, chunk_to_id, num_steps-1, 2)

    word_data_test = _file_to_word_ids(test_path, word_to_id, num_steps-1)
    pos_data_test = _file_to_tag_classifications(test_path, pos_to_id, num_steps-1, 1)
    chunk_data_test = _file_to_tag_classifications(test_path, chunk_to_id, num_steps-1, 2)

    return word_data_t, pos_data_t, chunk_data_t, word_data_v, \
        pos_data_v, chunk_data_v, word_to_id, pos_to_id, chunk_to_id, \
        word_data_test, pos_data_test, chunk_data_test, word_data_c, \
        pos_data_c, chunk_data_c, ptb_word_data, ptb_pos_data, ptb_chunk_data, word_embedding


"""
============
Section 2.
============
Here we want to feed in the raw data, batch-size, and window size
 and get back mini batches. These will be of size [batch_size, num_steps]
Args:
raw_words = the raw array of the word integers
raw_tag = the raw array of the tag integers
batch_size = batch size
num_steps = the number of steps you are going to look back in your rnn
tag_vocab_size = the size of the the number of tag tokens (
    needed for transfer into the [0,...,1,...,0] format)
Yields
(x,y) - x the batch, y the tag tags
"""


def create_batches(raw_words, raw_pos, raw_chunk, batch_size, num_steps, pos_vocab_size,
                   chunk_vocab_size, vocab_size, continuing=False):
    """Create those minibatches."""

    def _reshape_and_pad(tokens, batch_size, num_steps):
        tokens = np.array(tokens, dtype=np.int32)
        data_len = len(tokens)
        post_padding_required = (batch_size*num_steps) - np.mod(data_len, batch_size*num_steps)

        tokens = np.pad(tokens, (0, post_padding_required), 'constant',
                        constant_values=0)
        epoch_length = len(tokens) // (batch_size*num_steps)
        tokens = tokens.reshape([batch_size, num_steps*epoch_length])
        return tokens

    """
    1. Prepare the input (word) data
    """
    word_data = _reshape_and_pad(raw_words, batch_size, num_steps)
    pos_data = _reshape_and_pad(raw_pos, batch_size, num_steps)
    chunk_data = _reshape_and_pad(raw_chunk, batch_size, num_steps)

    """
    3. Do the epoch thing and iterate
    """
    data_len = len(raw_words)

    # how many times do you iterate to reach the end of the epoch
    epoch_size = (data_len // (batch_size*num_steps)) + 1

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    if continuing == False:
        for i in range(epoch_size):
            x = word_data[:, i*num_steps:(i+1)*num_steps]
            y_pos = np.vstack(_seq_tag(pos_data[tag, i*num_steps:(i+1)*num_steps],
                              pos_vocab_size) for tag in range(batch_size))
            y_chunk = np.vstack(_seq_tag(chunk_data[tag, i*num_steps:(i+1)*num_steps],
                                chunk_vocab_size) for tag in range(batch_size))
            y_lm = np.vstack(_seq_tag(word_data[tag, i*num_steps+1:(i+1)*num_steps+1],
                                vocab_size) for tag in range(batch_size))

            # append for last batch for lm
            if i == epoch_size-1:
                y_lm = np.vstack((y_lm,_seq_tag(np.zeros((batch_size,num_steps),dtype=np.int),vocab_size)))
            y_pos = y_pos.astype(np.int32)
            y_chunk = y_chunk.astype(np.int32)
            y_lm = y_lm.astype(np.int32)
            yield (x, y_pos, y_chunk, y_lm)
    else:
        i = 0
        while i > -1:
            x = word_data[:, i*num_steps:(i+1)*num_steps]
            y_pos = np.vstack(_seq_tag(pos_data[tag, i*num_steps:(i+1)*num_steps],
                              pos_vocab_size) for tag in range(batch_size))
            y_chunk = np.vstack(_seq_tag(chunk_data[tag, i*num_steps:(i+1)*num_steps],
                                chunk_vocab_size) for tag in range(batch_size))
            y_lm = np.vstack(_seq_tag(word_data[tag, i*num_steps+1:(i+1)*num_steps+1],
                                vocab_size) for tag in range(batch_size))

            # append for last batch for lm
            if i == epoch_size-1:
                y_lm = np.vstack((y_lm,_seq_tag(np.zeros((batch_size,num_steps),dtype=np.int),vocab_size)))
            y_pos = y_pos.astype(np.int32)
            y_chunk = y_chunk.astype(np.int32)
            y_lm = y_lm.astype(np.int32)
            i = (i+1) % (epoch_size - 1)
            yield (x, y_pos, y_chunk, y_lm)


def _res_to_list(res, batch_size, num_steps, to_id, w_length, to_str=False):

    tmp = np.concatenate([x.reshape(batch_size, num_steps)
                          for x in res], axis=1).reshape(-1)
    inv_dict = {v: k for k, v in to_id.items()}
    if to_str:
        result = np.array([inv_dict[x] for x in tmp])
    return result[range(num_steps-1, w_length)].reshape(-1,1)
