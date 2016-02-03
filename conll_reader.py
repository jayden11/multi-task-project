"""Conll Reader"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
import pdb

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import csv


input = open("../data/train.txt", 'rb')
output = open("../data/train_no_empty_rows.txt", 'wb')
writer = csv.writer(output, lineterminator='\n')
for row in csv.reader(input, delimiter=" "):
    if row:
        writer.writerow(row)
input.close()
output.close()

filename_queue = tf.train.string_input_producer(["../data/train_no_empty_rows.txt"])
# train_no_empty_rows

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


record_defaults = [tf.constant(['p'], dtype=tf.string),    # Column 0
                   tf.constant(['p'], dtype=tf.string),    # Column 1
                   tf.constant(['p'], dtype=tf.string)]

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
#record_defaults = [["p"], ["p"], ["p"]]

col1, col2, col3 = tf.decode_csv(
    value, record_defaults=record_defaults)

features = tf.pack([col2, col3])
with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col1])

  coord.request_stop()
  coord.join(threads)
