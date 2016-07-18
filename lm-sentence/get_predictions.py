from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.python.platform

import model_reader as reader
import numpy as np
import pdb
#import pandas as pd
from graph import Shared_Model
from run_epoch import run_epoch
import argparse
import saveload
import run_epoch_random
import time


class Config(object):
    def __init__(self, num_steps, encoder_size, pos_decoder_size, chunk_decoder_size,
    dropout, batch_size, pos_embedding_size, num_shared_layers, num_private_layers, chunk_embedding_size,
    lm_decoder_size, bidirectional, lstm, mix_percent, max_epoch):
        """Configuration for the network"""
        self.init_scale = 0.1 # initialisation scale
        self.learning_rate = 0.001 # learning_rate (if you are using SGD)
        self.max_grad_norm = 5 # for gradient clipping
        self.num_steps = int(num_steps) # length of sequence
        self.word_embedding_size = 300 # size of the embedding (consistent with glove)
        self.encoder_size = int(encoder_size) # first layer
        self.pos_decoder_size = int(pos_decoder_size) # second layer
        self.chunk_decoder_size = int(chunk_decoder_size) # second layer
        self.lm_decoder_size = int(lm_decoder_size) # second layer
        self.max_epoch = int(max_epoch) # maximum number of epochs
        self.keep_prob = float(dropout) # for dropout
        self.batch_size = int(batch_size) # number of sequence
        self.pos_embedding_size = int(pos_embedding_size)
        self.num_shared_layers = int(num_shared_layers)
        self.num_private_layers = int(num_private_layers)
        self.argmax = 0
        self.chunk_embedding_size = int(chunk_embedding_size)
        self.lm_decoder_size = int(lm_decoder_size)
        self.random_mix = True
        self.ptb = True
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.mix_percent = mix_percent

def main(model_type, dataset_path, ptb_path, save_path,
    num_steps, encoder_size, pos_decoder_size, chunk_decoder_size, dropout,
    batch_size, pos_embedding_size, num_shared_layers, num_private_layers, chunk_embedding_size,
    lm_decoder_size, bidirectional, lstm, write_to_file, mix_percent,glove_path,max_epoch,embedding=False):

    """Main."""
    config = Config(num_steps, encoder_size, pos_decoder_size, chunk_decoder_size, dropout,
    batch_size, pos_embedding_size, num_shared_layers, num_private_layers, chunk_embedding_size,
    lm_decoder_size, bidirectional, lstm, mix_percent, max_epoch)

    raw_data_path = dataset_path + '/data'
    raw_data = reader.raw_x_y_data(
        raw_data_path, config.num_steps, ptb_path + '/data', embedding, glove_path)

    words_t, pos_t, chunk_t, words_v, \
        pos_v, chunk_v, word_to_id, pos_to_id, \
        chunk_to_id, words_test, pos_test, chunk_test, \
        words_c, pos_c, chunk_c, words_ptb, pos_ptb, chunk_ptb, word_embedding = raw_data

    num_pos_tags = len(pos_to_id)
    num_chunk_tags = len(chunk_to_id)
    vocab_size = len(word_to_id)

    if embedding==True:
        word_embedding = np.float32(word_embedding)
    else:
        pdb.set_trace()
        word_embedding = np.float32((np.random.rand(vocab_size, config.word_embedding_size)-0.5)*config.init_scale)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        # model to train hyperparameters on
        with tf.variable_scope("hyp_model", reuse=None, initializer=initializer):
            m = Shared_Model(is_training=True, config=config, num_pos_tags=num_pos_tags,
            num_chunk_tags=num_chunk_tags, vocab_size=vocab_size, word_embedding=word_embedding)

        with tf.variable_scope("hyp_model", reuse=True, initializer=initializer):
            mValid = Shared_Model(is_training=False, config=config, num_pos_tags=num_pos_tags,
            num_chunk_tags=num_chunk_tags, vocab_size=vocab_size, word_embedding=word_embedding)

        # model that trains, given hyper-parameters
        with tf.variable_scope("final_model", reuse=None, initializer=initializer):
            mTrain = Shared_Model(is_training=True, config=config, num_pos_tags=num_pos_tags,
            num_chunk_tags=num_chunk_tags, vocab_size=vocab_size, word_embedding=word_embedding)

        with tf.variable_scope("final_model", reuse=True, initializer=initializer):
            mTest = Shared_Model(is_training=False, config=config, num_pos_tags=num_pos_tags,
            num_chunk_tags=num_chunk_tags, vocab_size=vocab_size, word_embedding=word_embedding)

        tf.initialize_all_variables().run()

        v_dict = saveload.load_np('../../data/outputs/temp/fin-model.pkl',session)
        for key, value in v_dict.items():
            try:
                session.run(tf.assign(v_dict[key], value))
            except:
                pdb.set_trace()

        print('getting training predictions')
        _, posp_t, chunkp_t, lmp_t, post_t, chunkt_t, lmt_t, _, _, _ = \
            run_epoch(session, mValid, words_t, pos_t, chunk_t,
                      num_pos_tags, num_chunk_tags, vocab_size,
                      verbose=True, valid=True, model_type=model_type)

        print('getting validation predictions')
        valid_loss, posp_v, chunkp_v, lmp_v, post_v, chunkt_v, lmt_v, pos_v_loss, chunk_v_loss, lm_v_loss = \
            run_epoch(session, mValid, words_v, pos_v, chunk_v,
                      num_pos_tags, num_chunk_tags, vocab_size,
                      verbose=True, valid=True, model_type=model_type)


        print('Getting Testing Predictions')
        _, posp_test, chunkp_test, _, _, _, _, _, _, _ = \
            run_epoch(session, mTest,
                      words_test, pos_test, chunk_test,
                      num_pos_tags, num_chunk_tags, vocab_size,
                      verbose=True, valid=True, model_type=model_type)


        print('Writing Predictions')


        # get training predictions as list
        posp_t = reader._res_to_list(posp_t, config.batch_size, config.num_steps,
                                     pos_to_id, len(words_t), to_str=True)
        chunkp_t = reader._res_to_list(chunkp_t, config.batch_size,
                                       config.num_steps, chunk_to_id, len(words_t), to_str=True)
        lmp_t = reader._res_to_list(lmp_t, config.batch_size,
                                        config.num_steps, word_to_id, len(words_t), to_str=True)
        post_t = reader._res_to_list(post_t, config.batch_size, config.num_steps,
                                     pos_to_id, len(words_t), to_str=True)
        chunkt_t = reader._res_to_list(chunkt_t, config.batch_size,
                                       config.num_steps, chunk_to_id, len(words_t), to_str=True)
        lmt_t = reader._res_to_list(lmt_t, config.batch_size,
                                        config.num_steps, word_to_id, len(words_t), to_str=True)


        # get predictions as list
        posp_v = reader._res_to_list(posp_v, config.batch_size, config.num_steps,
                                     pos_to_id, len(words_v), to_str=True)
        chunkp_v = reader._res_to_list(chunkp_v, config.batch_size,
                                       config.num_steps, chunk_to_id, len(words_v), to_str=True)
        lmp_v = reader._res_to_list(lmp_v, config.batch_size,
                                       config.num_steps, word_to_id, len(words_v), to_str=True)
        chunkt_v = reader._res_to_list(chunkt_v, config.batch_size,
                                       config.num_steps, chunk_to_id, len(words_v), to_str=True)
        post_v = reader._res_to_list(post_v, config.batch_size, config.num_steps,
                                     pos_to_id, len(words_v), to_str=True)
        lmt_v = reader._res_to_list(lmt_v, config.batch_size,
                                       config.num_steps, word_to_id, len(words_v), to_str=True)
        # prediction reshaping
        posp_c = reader._res_to_list(posp_c, config.batch_size, config.num_steps,
                                     pos_to_id, len(words_c),to_str=True)
        posp_test = reader._res_to_list(posp_test, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_test),to_str=True)
        chunkp_c = reader._res_to_list(chunkp_c, config.batch_size,
                                       config.num_steps, chunk_to_id, len(words_c),to_str=True)
        chunkp_test = reader._res_to_list(chunkp_test, config.batch_size, config.num_steps,
                                          chunk_to_id, len(words_test), to_str=True)


        train_custom = reader.read_tokens(raw_data_path + '/train.txt', 0,-1)
        valid_custom = reader.read_tokens(raw_data_path + '/validation.txt',0, -1)
        combined = reader.read_tokens(raw_data_path + '/train_val_combined.txt',0, -1)
        test_data = reader.read_tokens(raw_data_path + '/test.txt',0, -1)

        print('loaded text')
        chunk_pred_train = np.concatenate((np.transpose(train_custom), [str(s).upper() for s in chunkp_t]), axis=1)
        chunk_pred_val = np.concatenate((np.transpose(valid_custom), [str(s).upper() for s in chunkp_v]), axis=1)
        chunk_pred_c = np.concatenate((np.transpose(combined), [str(s).upper() for s in chunkp_c]), axis=1)
        chunk_pred_test = np.concatenate((np.transpose(test_data), [str(s).upper() for s in chunkp_test]), axis=1)
        pos_pred_train = np.concatenate((np.transpose(train_custom), [str(s).upper() for s in posp_t]), axis=1)
        pos_pred_val = np.concatenate((np.transpose(valid_custom), [str(s).upper() for s in posp_v]), axis=1)
        pos_pred_c = np.concatenate((np.transpose(combined), [str(s).upper for s in posp_c]), axis=1)
        pos_pred_test = np.concatenate((np.transpose(test_data), [str(s).upper() for s in posp_test]), axis=1)

        print('finished concatenating, about to start saving')

        np.savetxt(save_path + '/predictions/chunk_pred_train.txt',
                   chunk_pred_train, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_train.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_val.txt',
                   chunk_pred_val, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_combined.txt',
                   chunk_pred_c, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_test.txt',
                   chunk_pred_test, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_train.txt',
                   pos_pred_train, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_val.txt',
                   pos_pred_val, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_combined.txt',
                   pos_pred_c, fmt='%s')
        np.savetxt(save_path + '/predictions/pos_pred_test.txt',
                   pos_pred_test, fmt='%s')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type")
    parser.add_argument("--dataset_path")
    parser.add_argument("--ptb_path")
    parser.add_argument("--glove_path")
    parser.add_argument("--save_path")
    parser.add_argument("--num_steps")
    parser.add_argument("--encoder_size")
    parser.add_argument("--pos_decoder_size")
    parser.add_argument("--chunk_decoder_size")
    parser.add_argument("--dropout")
    parser.add_argument("--batch_size")
    parser.add_argument("--pos_embedding_size")
    parser.add_argument("--num_shared_layers")
    parser.add_argument("--num_private_layers")
    parser.add_argument("--chunk_embedding_size")
    parser.add_argument("--lm_decoder_size")
    parser.add_argument("--bidirectional")
    parser.add_argument("--lstm")
    parser.add_argument("--mix_percent")
    parser.add_argument("--write_to_file")
    parser.add_argument("--embedding")
    parser.add_argument("--max_epoch")
    args = parser.parse_args()
    if (str(args.model_type) != "POS") and (str(args.model_type) != "CHUNK"):
        args.model_type = 'JOINT'
    print('Model Selected : ' + str(args.model_type))
    main(str(args.model_type),str(args.dataset_path), \
         str(args.ptb_path),str(args.save_path), \
         int(args.num_steps), int(args.encoder_size), \
         int(args.pos_decoder_size), int(args.chunk_decoder_size), \
         float(args.dropout), int(args.batch_size), \
         int(args.pos_embedding_size), int(args.num_shared_layers), int(args.num_private_layers), \
         int(args.chunk_embedding_size), int(args.lm_decoder_size), \
         int(args.bidirectional), int(args.lstm), int(args.write_to_file), float(args.mix_percent), \
         str(args.glove_path), int(args.max_epoch),int(args.embedding))
