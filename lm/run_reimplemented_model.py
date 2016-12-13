from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.python.platform

import lm_model_reader as reader
import numpy as np
import pdb
#import pandas as pd
from reimplemented_model import ReimplementedModel
import argparse
import saveload
import run_epoch_random_reimplemented
import time
from sklearn.metrics import f1_score
import pickle


class Config(object):
    def __init__(self, num_steps, word_embedding_size, max_epoch, keep_prob,
            batch_size, mix_percent,layer_size, connection_embedding_size):
        """Configuration for the network"""
        self.init_scale = 0.1 # initialisation scale
        self.learning_rate = 0.1 # learning_rate (if you are using SGD)
        self.max_grad_norm = 5 # for gradient clipping
        self.num_steps = int(num_steps) # length of sequence
        self.word_embedding_size = word_embedding_size # size of the embedding (consistent with glove)
        self.max_epoch = int(max_epoch) # maximum number of epochs
        self.keep_prob = float(keep_prob) # for dropout
        self.batch_size = int(batch_size) # number of sequence
        self.random_mix = True
        self.ptb = True
        self.mix_percent = mix_percent
        self.num_steps = num_steps
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.word_embedding_size = word_embedding_size
        self.connection_embedding_size = connection_embedding_size
        self.embedding_trainable = True

def main(model_type, dataset_path, ptb_path, save_path, glove_path, num_steps,
        word_embedding_size, max_epoch, keep_prob, batch_size,
        mix_percent, layer_size, connection_embedding_size, test=False, embedding=True,
        write_to_file=True, fraction_of_training_data=1.0):

    """Main."""
    config = Config(num_steps, word_embedding_size, max_epoch, keep_prob,
            batch_size, mix_percent, layer_size,
            connection_embedding_size)

    raw_data_path = dataset_path + '/data'
    raw_data = reader.raw_x_y_data(
        raw_data_path, num_steps, ptb_path + '/data', True, glove_path, ptb=False)

    words_t, pos_t, chunk_t, words_v, \
        pos_v, chunk_v, word_to_id, pos_to_id, \
        chunk_to_id, words_test, pos_test, chunk_test, \
        words_c, pos_c, chunk_c, words_ptb, pos_ptb, chunk_ptb, word_embedding = raw_data

    num_train_examples = int(np.floor(len(words_t) * fraction_of_training_data))

    words_t = words_t[:num_train_examples]
    pos_t = pos_t[:num_train_examples]
    chunk_t = chunk_t[:num_train_examples]

    num_pos_tags = len(pos_to_id)
    num_chunk_tags = len(chunk_to_id)
    vocab_size = len(word_to_id)
    prev_chunk_F1 = 0.0

    ptb_batches = reader.create_batches(words_ptb, pos_ptb, chunk_ptb, config.batch_size,
                            config.num_steps, num_pos_tags, num_chunk_tags, vocab_size, continuing=True)

    ptb_iter = 0

    # Create an empty array to hold [epoch number, F1]
    if test==False:
        best_chunk_epoch = [0, 0.0]
        best_pos_epoch = [0, 0.0]
    else:
        best_chunk_epoch = [max_epoch, 0.0]

    print('constructing word embedding')

    if embedding==True:
        word_embedding = np.float32(word_embedding)
    else:
        word_embedding = np.float32((np.random.rand(vocab_size, config.word_embedding_size)-0.5)*config.init_scale)

    if test==False:
        with tf.Graph().as_default(), tf.Session() as session:
            print('building models')
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            # model to train hyperparameters on
            with tf.variable_scope("hyp_model", reuse=None, initializer=initializer):
                m = ReimplementedModel(is_training=True, config=config, num_pos_tags=num_pos_tags,
                num_chunk_tags=num_chunk_tags, vocab_size=vocab_size,
                word_embedding=word_embedding)

            with tf.variable_scope("hyp_model", reuse=True, initializer=initializer):
                mValid = ReimplementedModel(is_training=False, config=config, num_pos_tags=num_pos_tags,
                num_chunk_tags=num_chunk_tags, vocab_size=vocab_size,
                word_embedding=word_embedding)


            print('initialising variables')

            tf.initialize_all_variables().run()

            print("initialise word vectors")
            session.run(m.embedding_init, {m.embedding_placeholder: word_embedding})
            session.run(mValid.embedding_init, {mValid.embedding_placeholder: word_embedding})

            print('finding best epoch parameter')
            # ====================================
            # Create vectors for training results
            # ====================================

            # Create empty vectors for loss
            train_loss_stats = np.array([])
            train_pos_loss_stats = np.array([])
            train_chunk_loss_stats = np.array([])
            train_lm_loss_stats = np.array([])

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
            valid_lm_loss_stats = np.array([])

            # Create empty vectors for accuracy
            valid_pos_stats = np.array([])
            valid_chunk_stats = np.array([])

            for i in range(config.max_epoch):
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

                print("Epoch: %d" % (i + 1))
                if config.random_mix == False:
                    if config.ptb == True:
                        _, _, _, _, _, _, _, _, _, _ = \
                            run_epoch(session, m,
                                      words_ptb, pos_ptb, chunk_ptb,
                                      num_pos_tags, num_chunk_tags, vocab_size, num_steps,
                                      verbose=True, model_type='LM')


                    mean_loss, posp_t, chunkp_t, lmp_t, post_t, chunkt_t, lmt_t, pos_loss, chunk_loss, lm_loss = \
                        run_epoch(session, m,
                                  words_t, pos_t, chunk_t,
                                  num_pos_tags, num_chunk_tags, vocab_size, num_steps,
                                  verbose=True, model_type=model_type)

                else:
                    mean_loss, posp_t, chunkp_t, lmp_t, post_t, chunkt_t, lmt_t, pos_loss, chunk_loss, lm_loss, ptb_iter = \
                        run_epoch_random_reimplemented.run_epoch(session, m,
                                  words_t, words_ptb, pos_t, pos_ptb, chunk_t, chunk_ptb,
                                  num_pos_tags, num_chunk_tags, vocab_size, num_steps, config,
                                  ptb_batches, ptb_iter, verbose=True, model_type=model_type)


                print('epoch finished')
                # Save stats for charts
                train_loss_stats = np.append(train_loss_stats, mean_loss)
                train_pos_loss_stats = np.append(train_pos_loss_stats, pos_loss)
                train_chunk_loss_stats = np.append(train_chunk_loss_stats, chunk_loss)
                train_lm_loss_stats = np.append(train_lm_loss_stats, lm_loss)

                # get training predictions as list
                posp_t = reader._res_to_list(posp_t, config.batch_size, num_steps,
                                             pos_to_id, len(words_t), to_str=True)
                chunkp_t = reader._res_to_list(chunkp_t, config.batch_size, num_steps,
                                               chunk_to_id, len(words_t),to_str=True)
                lmp_t = reader._res_to_list(lmp_t, config.batch_size, num_steps,
                                                 word_to_id, len(words_t),to_str=True)
                post_t = reader._res_to_list(post_t, config.batch_size, num_steps,
                                             pos_to_id, len(words_t), to_str=True)
                chunkt_t = reader._res_to_list(chunkt_t, config.batch_size, num_steps,
                                                chunk_to_id, len(words_t), to_str=True)
                lmt_t = reader._res_to_list(lmt_t, config.batch_size, num_steps,
                                                 word_to_id, len(words_t),to_str=True)

                # find the accuracy
                print('finding accuracy')
                pos_acc = np.sum(posp_t==post_t)/float(len(posp_t))
                chunk_F1 = f1_score(chunkt_t, chunkp_t,average="weighted")

                # add to array
                train_pos_stats = np.append(train_pos_stats, pos_acc)
                train_chunk_stats = np.append(train_chunk_stats, chunk_F1)

                # print for tracking
                print("Pos Training Accuracy After Epoch %d :  %3f" % (i+1, pos_acc))
                print("Chunk Training F1 After Epoch %d : %3f" % (i+1, chunk_F1))

                valid_loss, posp_v, chunkp_v, lmp_v, post_v, chunkt_v, lmt_v, pos_v_loss, chunk_v_loss, lm_v_loss, ptb_iter = \
                    run_epoch_random_reimplemented.run_epoch(session, mValid,
                              words_v, words_ptb, pos_v, pos_ptb, chunk_v, chunk_ptb,
                              num_pos_tags, num_chunk_tags, vocab_size, num_steps, config,
                              ptb_batches, ptb_iter, verbose=True,  model_type=model_type, valid=True)

                # Save loss for charts
                valid_loss_stats = np.append(valid_loss_stats, valid_loss)
                valid_pos_loss_stats = np.append(valid_pos_loss_stats, pos_v_loss)
                valid_chunk_loss_stats = np.append(valid_chunk_loss_stats, chunk_v_loss)
                valid_lm_loss_stats = np.append(valid_lm_loss_stats, lm_v_loss)

                # get predictions as list
                posp_v = reader._res_to_list(posp_v, config.batch_size, num_steps,
                                             pos_to_id, len(words_v), to_str=True)
                chunkp_v = reader._res_to_list(chunkp_v, config.batch_size, num_steps,
                                                chunk_to_id, len(words_v), to_str=True)
                lmp_v = reader._res_to_list(lmp_v, config.batch_size, num_steps,
                                                word_to_id, len(words_v), to_str=True)
                chunkt_v = reader._res_to_list(chunkt_v, config.batch_size, num_steps,
                                                chunk_to_id, len(words_v), to_str=True)
                post_v = reader._res_to_list(post_v, config.batch_size, num_steps,
                                             pos_to_id, len(words_v), to_str=True)
                lmt_v = reader._res_to_list(lmt_v, config.batch_size, num_steps,
                                                word_to_id, len(words_v), to_str=True)

                # find accuracy
                pos_acc = np.sum(posp_v==post_v)/float(len(posp_v))
                chunk_F1 = f1_score(chunkt_v, chunkp_v, average="weighted")


                print("Pos Validation Accuracy After Epoch %d :  %3f" % (i+1, pos_acc))
                print("Chunk Validation F1 After Epoch %d : %3f" % (i+1, chunk_F1))

                # add to stats
                valid_pos_stats = np.append(valid_pos_stats, pos_acc)
                valid_chunk_stats = np.append(valid_chunk_stats, chunk_F1)

                if (abs(chunk_F1-prev_chunk_F1))<=0.001:
                    config.learning_rate = 0.8*config.learning_rate
                    print("learning rate updated")

                # update best parameters
                if(chunk_F1 > best_chunk_epoch[1]) or (pos_acc > best_pos_epoch[1]):
                    if pos_acc > best_pos_epoch[1]:
                        best_pos_epoch = [i+1, pos_acc]
                    if chunk_F1 > best_chunk_epoch[1]:
                        best_chunk_epoch = [i+1, chunk_F1]

                    saveload.save(save_path + '/val_model.pkl', session)
                    with open(save_path + '/pos_to_id.pkl', "wb") as file:
                        pickle.dump(pos_to_id, file)
                    with open(save_path + '/chunk_to_id.pkl', "wb") as file:
                        pickle.dump(chunk_to_id, file)
                    print("Model saved in file: %s" % save_path)

                    if write_to_file==True:
                        id_to_word = {v: k for k, v in word_to_id.items()}

                        words_t_unrolled = [id_to_word[k] for k in words_t[num_steps-1:]]
                        words_v_unrolled = [id_to_word[k] for k in words_v[num_steps-1:]]

                        # unroll data
                        train_custom = np.hstack((np.array(words_t_unrolled).reshape(-1,1), np.char.upper(post_t), np.char.upper(chunkt_t)))
                        valid_custom = np.hstack((np.array(words_v_unrolled).reshape(-1,1), np.char.upper(post_v), np.char.upper(chunkt_v)))
                        chunk_pred_train = np.concatenate((train_custom, np.char.upper(chunkp_t).reshape(-1,1)), axis=1)
                        chunk_pred_val = np.concatenate((valid_custom, np.char.upper(chunkp_v).reshape(-1,1)), axis=1)
                        pos_pred_train = np.concatenate((train_custom, np.char.upper(posp_t).reshape(-1,1)), axis=1)
                        pos_pred_val = np.concatenate((valid_custom, np.char.upper(posp_v).reshape(-1,1)), axis=1)

                        # write to file
                        np.savetxt(save_path + '/predictions/chunk_pred_train.txt',
                                   chunk_pred_train, fmt='%s')
                        print('writing to ' + save_path + '/predictions/chunk_pred_train.txt')
                        np.savetxt(save_path + '/predictions/chunk_pred_val.txt',
                                   chunk_pred_val, fmt='%s')
                        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
                        np.savetxt(save_path + '/predictions/pos_pred_train.txt',
                                   pos_pred_train, fmt='%s')
                        print('writing to ' + save_path + '/predictions/pos_pred_train.txt')
                        np.savetxt(save_path + '/predictions/pos_pred_val.txt',
                                   pos_pred_val, fmt='%s')
                        print('writing to ' + save_path + '/predictions/pos_pred_val.txt')

                        print('Getting Testing Predictions (Valid)')
                        test_loss, posp_test, chunkp_test, lmp_test, post_test, chunkt_test, lmt_test, pos_test_loss, chunk_test_loss, lm_test_loss, ptb_iter = \
                            run_epoch_random_reimplemented.run_epoch(session, mValid,
                                      words_test, words_ptb, pos_test, pos_ptb, chunk_test, chunk_ptb,
                                      num_pos_tags, num_chunk_tags, vocab_size, num_steps, config,
                                      ptb_batches, ptb_iter, verbose=True,  model_type=model_type, valid=True)

                        # get predictions as list
                        posp_test = reader._res_to_list(posp_test, config.batch_size, num_steps,
                                                     pos_to_id, len(words_test), to_str=True)
                        chunkp_test = reader._res_to_list(chunkp_test, config.batch_size, num_steps,
                                                        chunk_to_id, len(words_test), to_str=True)
                        lmp_test = reader._res_to_list(lmp_test, config.batch_size, num_steps,
                                                        word_to_id, len(words_test), to_str=True)
                        chunkt_test = reader._res_to_list(chunkt_test, config.batch_size, num_steps,
                                                        chunk_to_id, len(words_test), to_str=True)
                        post_test = reader._res_to_list(post_test, config.batch_size, num_steps,
                                                     pos_to_id, len(words_test), to_str=True)
                        lmt_test = reader._res_to_list(lmt_test, config.batch_size, num_steps,
                                                        word_to_id, len(words_test), to_str=True)

                        words_test_c = [id_to_word[k] for k in words_test[num_steps-1:]]
                        test_data = np.hstack((np.array(words_test_c).reshape(-1,1), np.char.upper(post_test), np.char.upper(chunkt_test)))

                        # find the accuracy
                        print('finding  test accuracy')
                        pos_acc_train = np.sum(posp_test==post_test)/float(len(posp_test))
                        chunk_F1_train = f1_score(chunkt_test, chunkp_test,average="weighted")

                        print("POS Test Accuracy: " + str(pos_acc_train))
                        print("Chunk Test F1: " + str(chunk_F1_train))

                        chunk_pred_test = np.concatenate((test_data, np.char.upper(chunkp_test).reshape(-1,1)), axis=1)
                        pos_pred_test = np.concatenate((test_data, np.char.upper(posp_test).reshape(-1,1)), axis=1)

                        print('writing to ' + save_path + '/predictions/chunk_pred_combined.txt')
                        np.savetxt(save_path + '/predictions/chunk_pred_test.txt',
                                   chunk_pred_test, fmt='%s')
                        print('writing to ' + save_path + '/predictions/chunk_pred_test.txt')

                        np.savetxt(save_path + '/predictions/pos_pred_train.txt',
                                   pos_pred_train, fmt='%s')
                        print('writing to ' + save_path + '/predictions/pos_pred_train.txt')
                        np.savetxt(save_path + '/predictions/pos_pred_val.txt',
                                   pos_pred_val, fmt='%s')
                        print('writing to ' + save_path + '/predictions/pos_pred_val.txt')

                        np.savetxt(save_path + '/predictions/pos_pred_test.txt',
                                   pos_pred_test, fmt='%s')

                prev_chunk_F1 = chunk_F1

            # Save loss & accuracy plots
            np.savetxt(save_path + '/loss/valid_loss_stats.txt', valid_loss_stats)
            np.savetxt(save_path + '/loss/valid_pos_loss_stats.txt', valid_pos_loss_stats)
            np.savetxt(save_path + '/loss/valid_chunk_loss_stats.txt', valid_chunk_loss_stats)
            np.savetxt(save_path + '/accuracy/valid_pos_stats.txt', valid_pos_stats)
            np.savetxt(save_path + '/accuracy/valid_chunk_stats.txt', valid_chunk_stats)

            np.savetxt(save_path + '/loss/train_loss_stats.txt', train_loss_stats)
            np.savetxt(save_path + '/loss/train_pos_loss_stats.txt', train_pos_loss_stats)
            np.savetxt(save_path + '/loss/train_chunk_loss_stats.txt', train_chunk_loss_stats)
            np.savetxt(save_path + '/accuracy/train_pos_stats.txt', train_pos_stats)
            np.savetxt(save_path + '/accuracy/train_chunk_stats.txt', train_chunk_stats)

    if write_to_file == False:
            with tf.Graph().as_default(), tf.Session() as session:
                initializer = tf.random_uniform_initializer(-config.init_scale,
                                                            config.init_scale)

                with tf.variable_scope("final_model", reuse=None, initializer=initializer):
                    mTrain = ReimplementedModel(is_training=True, config=config, num_pos_tags=num_pos_tags,
                    num_chunk_tags=num_chunk_tags, vocab_size=vocab_size,
                    word_embedding=word_embedding)

                with tf.variable_scope("final_model", reuse=True, initializer=initializer):
                    mTest = ReimplementedModel(is_training=False, config=config, num_pos_tags=num_pos_tags,
                    num_chunk_tags=num_chunk_tags, vocab_size=vocab_size,
                    word_embedding=word_embedding)

                print("initialise variables")
                tf.initialize_all_variables().run()
                print("initialise word embeddings")
                session.run(mTrain.embedding_init, {mTrain.embedding_placeholder: word_embedding})
                session.run(mTest.embedding_init, {mTest.embedding_placeholder: word_embedding})

                # Train given epoch parameter
                if config.random_mix == False:
                    print('Train Given Best Epoch Parameter :' + str(best_chunk_epoch[0]))
                    for i in range(best_chunk_epoch[0]):
                        print("Epoch: %d" % (i + 1))
                        if config.ptb == False:
                            _, _, _, _, _, _, _, _, _, _ = \
                                run_epoch(session, mTrain,
                                          words_ptb, pos_ptb, chunk_ptb,
                                          num_pos_tags, num_chunk_tags, vocab_size, num_steps,
                                          verbose=True, model_type="LM")

                        _, posp_c, chunkp_c, _, _, _, _, _, _, _ = \
                            run_epoch(session, mTrain,
                                      words_c, pos_c, chunk_c,
                                      num_pos_tags, num_chunk_tags, vocab_size,
                                      verbose=True, model_type=model_type)

                else:
                    print('Train Given Best Epoch Parameter :' + str(best_chunk_epoch[0]))
                    for i in range(best_chunk_epoch[0]):
                        print("Epoch: %d" % (i + 1))
                        _, posp_c, chunkp_c, _, post_c, chunkt_c, _, _, _, _, ptb_iter = \
                            run_epoch_random_reimplemented.run_epoch(session, mTrain,
                                      words_c, words_ptb, pos_c, pos_ptb, chunk_c, chunk_ptb,
                                      num_pos_tags, num_chunk_tags, vocab_size, num_steps, config,
                                      ptb_batches, ptb_iter, verbose=True, model_type=model_type)


                print('Getting Testing Predictions')
                test_loss, posp_test, chunkp_test, lmp_test, post_test, chunkt_test, lmt_test, pos_test_loss, chunk_test_loss, lm_test_loss, ptb_iter = \
                    run_epoch_random_reimplemented.run_epoch(session, mTest,
                              words_test, words_ptb, pos_test, pos_ptb, chunk_test, chunk_ptb,
                              num_pos_tags, num_chunk_tags, vocab_size, num_steps, config,
                              ptb_batches, ptb_iter, verbose=True,  model_type=model_type, valid=True)

                print('Writing Predictions')
                # prediction reshaping
                posp_c = reader._res_to_list(posp_c, config.batch_size, num_steps,
                                             pos_to_id, len(words_c), to_str=True)
                posp_test = reader._res_to_list(posp_test, config.batch_size, num_steps,
                                                pos_to_id, len(words_test), to_str=True)
                chunkp_c = reader._res_to_list(chunkp_c, config.batch_size, num_steps,
                                               chunk_to_id, len(words_c),to_str=True)
                chunkp_test = reader._res_to_list(chunkp_test, config.batch_size, num_steps,
                                                  chunk_to_id, len(words_test),  to_str=True)

                post_c = reader._res_to_list(post_c, config.batch_size, num_steps,
                                             pos_to_id, len(words_c), to_str=True)
                post_test = reader._res_to_list(post_test, config.batch_size, num_steps,
                                                pos_to_id, len(words_test), to_str=True)
                chunkt_c = reader._res_to_list(chunkt_c, config.batch_size, num_steps,
                                               chunk_to_id, len(words_c),to_str=True)
                chunkt_test = reader._res_to_list(chunkt_test, config.batch_size, num_steps,
                                                  chunk_to_id, len(words_test),  to_str=True)

                # save pickle - save_path + '/saved_variables.pkl'
                print('saving checkpoint')
                saveload.save(save_path + '/fin_model.ckpt', session)

                words_t = [id_to_word[k] for k in words_t[num_steps-1:]]
                words_v = [id_to_word[k] for k in words_v[num_steps-1:]]
                words_c = [id_to_word[k] for k in words_c[num_steps-1:]]
                words_test = [id_to_word[k] for k in words_test[num_steps-1:]]

                # find the accuracy
                print('finding test accuracy')
                pos_acc = np.sum(posp_test==post_test)/float(len(posp_test))
                chunk_F1 = f1_score(chunkt_test, chunkp_test,average="weighted")

                print("POS Test Accuracy (Both): " + str(pos_acc))
                print("Chunk Test F1(Both): " + str(chunk_F1))

                print("POS Test Accuracy (Train): " + str(pos_acc_train))
                print("Chunk Test F1 (Train): " + str(chunk_F1_train))


                if test==False:
                    train_custom = np.hstack((np.array(words_t).reshape(-1,1), np.char.upper(post_t), np.char.upper(chunkt_t)))
                    valid_custom = np.hstack((np.array(words_v).reshape(-1,1), np.char.upper(post_v), np.char.upper(chunkt_v)))
                combined = np.hstack((np.array(words_c).reshape(-1,1), np.char.upper(post_c), np.char.upper(chunkt_c)))
                test_data = np.hstack((np.array(words_test).reshape(-1,1), np.char.upper(post_test), np.char.upper(chunkt_test)))

                print('loaded text')

                if test==False:
                    chunk_pred_train = np.concatenate((train_custom, np.char.upper(chunkp_t).reshape(-1,1)), axis=1)
                    chunk_pred_val = np.concatenate((valid_custom, np.char.upper(chunkp_v).reshape(-1,1)), axis=1)
                chunk_pred_c = np.concatenate((combined, np.char.upper(chunkp_c).reshape(-1,1)), axis=1)
                chunk_pred_test = np.concatenate((test_data, np.char.upper(chunkp_test).reshape(-1,1)), axis=1)
                if test==False:
                    pos_pred_train = np.concatenate((train_custom, np.char.upper(posp_t).reshape(-1,1)), axis=1)
                    pos_pred_val = np.concatenate((valid_custom, np.char.upper(posp_v).reshape(-1,1)), axis=1)
                pos_pred_c = np.concatenate((combined, np.char.upper(posp_c).reshape(-1,1)), axis=1)
                pos_pred_test = np.concatenate((test_data, np.char.upper(posp_test).reshape(-1,1)), axis=1)

                print('finished concatenating, about to start saving')

                if test == False:
                    np.savetxt(save_path + '/predictions/chunk_pred_train.txt',
                               chunk_pred_train, fmt='%s')
                    print('writing to ' + save_path + '/predictions/chunk_pred_train.txt')
                    np.savetxt(save_path + '/predictions/chunk_pred_val.txt',
                               chunk_pred_val, fmt='%s')
                    print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')

                np.savetxt(save_path + '/predictions/chunk_pred_combined.txt',
                           chunk_pred_c, fmt='%s')
                print('writing to ' + save_path + '/predictions/chunk_pred_combined.txt')
                np.savetxt(save_path + '/predictions/chunk_pred_test.txt',
                           chunk_pred_test, fmt='%s')
                print('writing to ' + save_path + '/predictions/chunk_pred_test.txt')

                if test == False:
                    np.savetxt(save_path + '/predictions/pos_pred_train.txt',
                               pos_pred_train, fmt='%s')
                    print('writing to ' + save_path + '/predictions/pos_pred_train.txt')
                    np.savetxt(save_path + '/predictions/pos_pred_val.txt',
                               pos_pred_val, fmt='%s')
                    print('writing to ' + save_path + '/predictions/pos_pred_val.txt')

                np.savetxt(save_path + '/predictions/pos_pred_combined.txt',
                           pos_pred_c, fmt='%s')
                np.savetxt(save_path + '/predictions/pos_pred_test.txt',
                           pos_pred_test, fmt='%s')

    else:
        print('Best Validation F1 ' + str(best_chunk_epoch[1]))
        print('Best Validation Epoch ' + str(best_chunk_epoch[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type")
    parser.add_argument("--dataset_path")
    parser.add_argument("--ptb_path")
    parser.add_argument("--glove_path")
    parser.add_argument("--save_path")
    parser.add_argument("--num_steps")
    parser.add_argument("--layer_size")
    parser.add_argument("--dropout")
    parser.add_argument("--batch_size")
    parser.add_argument("--connection_embedding_size")
    parser.add_argument("--mix_percent")
    parser.add_argument("--max_epoch")
    parser.add_argument("--word_embedding_size")
    parser.add_argument("--fraction_of_training_data")

    args = parser.parse_args()
    if (str(args.model_type) != "POS") and (str(args.model_type) != "CHUNK"):
        args.model_type = 'JOINT'
    print('Model Selected : ' + str(args.model_type))
    main(str(args.model_type),str(args.dataset_path),
         str(args.ptb_path),str(args.save_path), str(args.glove_path),
         int(args.num_steps), int(args.word_embedding_size), int(args.max_epoch),
         float(args.dropout), int(args.batch_size),
         float(args.mix_percent), int(args.layer_size), int(args.connection_embedding_size),
        fraction_of_training_data=float(args.fraction_of_training_data))
