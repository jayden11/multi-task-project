from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.python.platform

import model_reader as reader
import numpy as np
import pdb
import pandas as pd
from graph import Shared_Model
from run_epoch import run_epoch
import argparse


class Config(object):
    """Configuration for the network"""
    init_scale = 0.1
    learning_rate = 0.001
    max_grad_norm = 5
    num_steps = 20
    encoder_size = 200
    pos_decoder_size = 200
    chunk_decoder_size = 200
    max_epoch = 50
    keep_prob = 0.5
    batch_size = 64
    vocab_size = 20000
    num_pos_tags = 45
    num_chunk_tags = 23

def main(model_type):
    """Main."""
    config = Config()
    raw_data = reader.raw_x_y_data(
        '/Users/jonathangodwin/project/Conll/data/', config.num_steps)
    words_t, pos_t, chunk_t, words_v, \
        pos_v, chunk_v, word_to_id, pos_to_id, \
        chunk_to_id, words_test, pos_test, chunk_test, \
        words_c, pos_c, chunk_c = raw_data

    config.num_pos_tags = len(pos_to_id)
    config.num_chunk_tags = len(chunk_to_id)


    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.variable_scope("hyp_model", reuse=None, initializer=initializer):
            m = Shared_Model(is_training=True, config=config)
        with tf.variable_scope("hyp_model", reuse=True, initializer=initializer):
            mvalid = Shared_Model(is_training=False, config=config)
        with tf.variable_scope("fin_model", reuse=None, initializer=initializer):
            mTrain = Shared_Model(is_training=True, config=config)
        with tf.variable_scope("fin_model", reuse=True, initializer=initializer):
            mTest = Shared_Model(is_training=False, config=config)

        tf.initialize_all_variables().run()

        # Create an empty array to hold [epoch number, loss]
        best_epoch = [0, 100000]

        print('finding best epoch parameter')
        # ====================================
        # Create vectors for training results
        # ====================================

        # Create empty vectors for loss
        train_loss_stats = np.array([])
        train_pos_loss_stats = np.array([])
        train_chunk_loss_stats = np.array([])
        # Create empty vectors for accuracy
        train_pos_stats = np.array([])
        train_chunk_stats = np.array([])

        # ====================================
        # Create vectors for validation results
        # ====================================
        # Create empty vectors for loss
        valid_loss_stats = np.array([])
        valid_pos_loss_stats = np.array([])
        valid_chunk_loss_stats = np.array([])
        # Create empty vectors for accuracy
        valid_pos_stats = np.array([])
        valid_chunk_stats = np.array([])

        for i in range(config.max_epoch):
            print("Epoch: %d" % (i + 1))
            mean_loss, posp_t, chunkp_t, post_t, chunkt_t, pos_loss, chunk_loss = \
                run_epoch(session, m,
                          words_t, pos_t, chunk_t,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, model_type=model_type)

            # Save stats for charts
            train_loss_stats = np.append(train_loss_stats, mean_loss)
            train_pos_loss_stats = np.append(train_pos_loss_stats, pos_loss)
            train_chunk_loss_stats = np.append(train_chunk_loss_stats, chunk_loss)

            # get predictions as list
            posp_t = reader._res_to_list(posp_t, config.batch_size, config.num_steps,
                                         pos_to_id, len(words_t))
            chunkp_t = reader._res_to_list(chunkp_t, config.batch_size,
                                           config.num_steps, chunk_to_id, len(words_t))
            post_t = reader._res_to_list(post_t, config.batch_size, config.num_steps,
                                         pos_to_id, len(words_t))
            chunkt_t = reader._res_to_list(chunkt_t, config.batch_size,
                                           config.num_steps, chunk_to_id, len(words_t))

            # find the accuracy
            pos_acc = np.sum(posp_t == post_t)/float(len(posp_t))
            chunk_acc = np.sum(chunkp_t == chunkt_t)/float(len(chunkp_t))

            # write to file
            train_pos_stats = np.append(train_pos_stats, pos_acc)
            train_chunk_stats = np.append(train_chunk_stats, chunk_acc)

            # print for tracking
            print("Pos Training Accuracy After Epoch %d :  %3f" % (i+1, pos_acc))
            print("Chunk Training Accuracy After Epoch %d : %3f" % (i+1, chunk_acc))

            valid_loss, posp_v, chunkp_v, post_v, chunkt_v, pos_v_loss, chunk_v_loss = \
                run_epoch(session, mvalid, words_v, pos_v, chunk_v,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, valid=True, model_type=model_type)

            # Save loss for charts
            valid_loss_stats = np.append(valid_loss_stats, valid_loss)
            valid_pos_loss_stats = np.append(valid_pos_loss_stats, pos_v_loss)
            valid_chunk_loss_stats = np.append(valid_chunk_loss_stats, chunk_v_loss)

            # get predictions as list

            posp_v = reader._res_to_list(posp_v, config.batch_size, config.num_steps,
                                         pos_to_id, len(words_v))
            chunkp_v = reader._res_to_list(chunkp_v, config.batch_size,
                                           config.num_steps, chunk_to_id, len(words_v))
            chunkt_v = reader._res_to_list(chunkt_v, config.batch_size,
                                           config.num_steps, chunk_to_id, len(words_v))
            post_v = reader._res_to_list(post_v, config.batch_size, config.num_steps,
                                         pos_to_id, len(words_v))

            # find accuracy
            pos_acc = np.sum(posp_v == post_v)/float(len(posp_v))
            chunk_acc = np.sum(chunkp_v == chunkt_v)/float(len(chunkp_v))

            print("Pos Validation Accuracy After Epoch %d :  %3f" % (i+1, pos_acc))
            print("Chunk Validation Accuracy After Epoch %d : %3f" % (i+1, chunk_acc))

            # write to file
            valid_pos_stats = np.append(valid_pos_stats, pos_acc)
            valid_chunk_stats = np.append(valid_chunk_stats, chunk_acc)

            # update best parameters
            if(valid_loss < best_epoch[1]):
                best_epoch = [i+1, valid_loss]

        # Save loss & accuracy plots
        np.savetxt('../../data/current_outcome/loss/valid_loss_stats.txt', valid_loss_stats)
        np.savetxt('../../data/current_outcome/loss/valid_pos_loss_stats.txt', valid_pos_loss_stats)
        np.savetxt('../../data/current_outcome/loss/valid_chunk_loss_stats.txt', valid_chunk_loss_stats)

        np.savetxt('../../data/current_outcome/accuracy/valid_pos_stats.txt', valid_pos_stats)
        np.savetxt('../../data/current_outcome/accuracy/valid_chunk_stats.txt', valid_chunk_stats)

        np.savetxt('../../data/current_outcome/loss/train_loss_stats.txt', train_loss_stats)
        np.savetxt('../../data/current_outcome/loss/train_pos_loss_stats.txt', train_pos_loss_stats)
        np.savetxt('../../data/current_outcome/loss/train_chunk_loss_stats.txt', train_chunk_loss_stats)
        np.savetxt('../../data/current_outcome/accuracy/train_pos_stats.txt', train_pos_stats)
        np.savetxt('../../data/current_outcome/accuracy/train_chunk_stats.txt', train_chunk_stats)

        # Train given epoch parameter
        print('Train Given Best Epoch Parameter :' + str(best_epoch[0]))
        for i in range(best_epoch[0]):
            print("Epoch: %d" % (i + 1))
            _, posp_c, chunkp_c, _, _, _, _ = \
                run_epoch(session, mTrain,
                          words_c, pos_c, chunk_c,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, model_type=model_type)

        print('Getting Testing Predictions')
        _, posp_test, chunkp_test, _, _, _, _ = \
            run_epoch(session, mTest,
                      words_test, pos_test, chunk_test,
                      config.num_pos_tags, config.num_chunk_tags,
                      verbose=True, valid=True, model_type=model_type)

        print('Writing Predictions')
        # prediction reshaping
        posp_c = reader._res_to_list(posp_c, config.batch_size, config.num_steps,
                                     pos_to_id, len(words_c))
        posp_test = reader._res_to_list(posp_test, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_test))
        chunkp_c = reader._res_to_list(chunkp_c, config.batch_size,
                                       config.num_steps, chunk_to_id, len(words_c))
        chunkp_test = reader._res_to_list(chunkp_test, config.batch_size, config.num_steps,
                                          chunk_to_id, len(words_test))




        print('saving')
        train_custom = pd.read_csv('../../data/train_custom.txt', sep= ' ',header=None).as_matrix()
        valid_custom = pd.read_csv('../../data/val_custom.txt', sep= ' ',header=None).as_matrix()
        combined = pd.read_csv('../../data/train.txt', sep= ' ',header=None).as_matrix()
        test_data = pd.read_csv('../../data/test.txt', sep= ' ',header=None).as_matrix()

        chunk_pred_train = np.concatenate((train_custom, chunkp_t), axis=1)
        chunk_pred_val = np.concatenate((valid_custom, chunkp_v), axis=1)
        chunk_pred_c = np.concatenate((combined, chunkp_c), axis=1)
        chunk_pred_test = np.concatenate((test_data, chunkp_test), axis=1)
        pos_pred_train = np.concatenate((train_custom, posp_t), axis=1)
        pos_pred_val = np.concatenate((valid_custom, posp_v), axis=1)
        pos_pred_c = np.concatenate((combined, posp_c), axis=1)
        pos_pred_test = np.concatenate((test_data, posp_test), axis=1)

        np.savetxt('../../data/current_outcome/predictions/chunk_pred_train.txt',
                   chunk_pred_train, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/chunk_pred_val.txt',
                   chunk_pred_val, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/chunk_pred_combined.txt',
                   chunk_pred_c, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/chunk_pred_test.txt',
                   chunk_pred_test, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/pos_pred_train.txt',
                   pos_pred_train, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/pos_pred_val.txt',
                   pos_pred_val, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/pos_pred_combined.txt',
                   pos_pred_c, fmt='%s')
        np.savetxt('../../data/current_outcome/predictions/pos_pred_test.txt',
                   pos_pred_test, fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type")
    args = parser.parse_args()
    if (str(args.model_type) != "POS") and (str(args.model_type) != "CHUNK"):
        args.model_type = 'JOINT'
    print('Model Selected : ' + str(args.model_type))
    main(str(args.model_type))
    #tf.app.run()
