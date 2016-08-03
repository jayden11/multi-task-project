from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


from subgraph.pos_graph import pos_private
from subgraph.chunk_graph import chunk_private
from subgraph.shared_graph import shared_layer
from subgraph.lm_graph import lm_private

import pdb

class Shared_Model(object):

    def __init__(self, config, is_training, num_pos_tags, num_chunk_tags,
        vocab_size, num_steps, embedding_dim, projection_size):
        """Initialisation
            basically set the self-variables up, so that we can call them
            as variables to the model.
        """
        self.max_grad_norm = max_grad_norm = config.max_grad_norm
        self.num_steps = num_steps
        self.encoder_size = encoder_size = config.encoder_size
        self.pos_decoder_size = pos_decoder_size = config.pos_decoder_size
        self.chunk_decoder_size = chunk_decoder_size = config.chunk_decoder_size
        self.batch_size = batch_size = config.batch_size
        self.vocab_size = vocab_size
        self.num_pos_tags = num_pos_tags
        self.num_chunk_tags = num_chunk_tags
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.pos_embedding_size = pos_embedding_size = config.pos_embedding_size
        self.chunk_embedding_size = chunk_embedding_size = config.chunk_embedding_size
        self.num_private_layers = num_private_layers = config.num_private_layers
        self.argmax = config.argmax
        self.lm_decoder_size = config.lm_decoder_size
        self.mix_percent = config.mix_percent

        # add input size - size of pos tags
        self.pos_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                          num_pos_tags])
        self.chunk_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                            num_chunk_tags])
        self.lm_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                            vocab_size])

        # create a regulariser
        self.gold_embed = tf.placeholder(tf.int32, shape=[], name="condition")

        def _loss(logits, labels):
            """Calculate loss for both pos and chunk
                Args:
                    logits from the decoder
                    labels - one-hot
                returns:
                    loss as tensor of type float
            """

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                    labels,
                                                                    name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            (_, int_targets) = tf.nn.top_k(labels, 1)
            (_, int_predictions) = tf.nn.top_k(logits, 1)
            num_true = tf.reduce_sum(tf.cast(tf.equal(int_targets, int_predictions), tf.float32))
            accuracy = num_true / (num_steps*batch_size)
            return loss, accuracy, int_predictions, int_targets

        def _training(loss, config, m):
            """Sets up training ops and also...

            Create a summarisor for tensorboard

            Creates the optimiser

            The op returned from this is what is passed to session run

                Args:
                    loss float
                    learning_rate float

                returns:

                Op for training
            """
            # Create the gradient descent optimizer with the
            # given learning rate.
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                              config.max_grad_norm)
            optimizer = tf.train.AdamOptimizer()
            if config.adam == True:
                optimizer = tf.train.AdamOptimizer()
            else:
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
            return train_op

        def input_projection3D(input3D, projection_size):

            hidden = input3D.get_shape()[2].value
            steps = input3D.get_shape()[1].value
            if hidden < projection_size : print("WARNING - projecting to higher dimension than original embeddings")
            inputs = tf.reshape(input3D, [-1, steps, 1, hidden]) # now shape (batch, num_steps, 1, hidden_size)
            W_proj = tf.get_variable("W_proj", [1,1,hidden, projection_size])
            b_proj = tf.get_variable("b_proj", [projection_size])

            projection = tf.nn.conv2d(inputs, W_proj, [1,1,1,1], "SAME")
            projection = tf.tanh(tf.nn.bias_add(projection,b_proj))
            return tf.reshape(projection, [-1, steps,projection_size])

        self.sentence_lengths = sentence_lengths =  tf.placeholder(tf.int32, [batch_size])
        self.learning_rate = learning_rate = tf.placeholder(tf.float32, [], "learning_rate")

        word_embedding = word_embedding = tf.get_variable("word_embedding", [vocab_size, embedding_dim], trainable=True)
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        self.embedding_init = word_embedding.assign(self.embedding_placeholder)

        inputs = tf.nn.embedding_lookup(word_embedding, self.input_data)
        if config.embedding_trainable == False:
            inputs = input_projection3D(inputs, projection_size) # put them through a projections
        input_l2 = tf.reduce_sum(inputs) # sum them up for the regulariser

        self.pos_embedding = pos_embedding = tf.get_variable("pos_embedding",
            [num_pos_tags, pos_embedding_size])
        self.chunk_embedding = chunk_embedding = tf.get_variable("chunk_embedding",
            [num_chunk_tags, chunk_embedding_size])

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        encoding = shared_layer(inputs, config, is_training, sentence_lengths)

        encoding = tf.pack(encoding)
        encoding = tf.transpose(encoding, perm=[1, 0, 2])
        encoding_l2 = tf.reduce_sum(encoding)

        pos_logits, pos_hidden, pos_l2 = pos_private(encoding, config, is_training, sentence_lengths)

        pos_loss, pos_accuracy, pos_int_pred, pos_int_targ = _loss(pos_logits, self.pos_targets)
        self.pos_loss = pos_loss

        # make the variables available to the outside
        self.pos_int_pred = pos_int_pred
        self.pos_int_targ = pos_int_targ

        # choose either argmax or dot product for pos embedding
        if config.argmax==1:
            pos_to_chunk_embed = tf.cond(self.gold_embed > 0 , lambda: tf.matmul(self.pos_targets, pos_embedding),\
            lambda: tf.nn.embedding_lookup(pos_embedding,pos_int_pred))
        else:
            pos_to_chunk_embed = tf.cond(self.gold_embed > 0 , lambda: tf.matmul(self.pos_targets, pos_embedding), \
            lambda: tf.matmul(tf.nn.softmax(pos_logits),pos_embedding))

        chunk_logits, chunk_hidden, chunk_l2 = chunk_private(encoding, pos_to_chunk_embed, config, is_training, sentence_lengths)

        chunk_loss, chunk_accuracy, chunk_int_pred, chunk_int_targ = _loss(chunk_logits, self.chunk_targets)

        self.chunk_loss = chunk_loss
        self.chunk_int_pred = chunk_int_pred
        self.chunk_int_targ = chunk_int_targ

        # choose either argmax or dot product for chunk embedding
        if config.argmax==1:
            chunk_to_lm_embed = tf.cond(self.gold_embed > 0, lambda: tf.matmul(self.chunk_targets, chunk_embedding), \
            lambda: tf.nn.embedding_lookup(chunk_embedding,chunk_int_pred))
        else:
            chunk_to_lm_embed = tf.cond(self.gold_embed > 0, lambda: tf.matmul(tf.nn.softmax(chunk_logits),chunk_embedding), \
            lambda: tf.nn.embedding_lookup(chunk_embedding,chunk_int_pred))

        lm_logits, lm_l2 = lm_private(encoding, chunk_to_lm_embed,
            pos_to_chunk_embed, config, is_training, sentence_lengths)

        lm_loss, lm_accuracy, lm_int_pred, lm_int_targ = _loss(lm_logits, self.lm_targets)

        self.lm_loss = lm_loss
        self.lm_int_pred = lm_int_pred
        self.lm_int_targ = lm_int_targ
        self.joint_loss = (chunk_loss + pos_loss + lm_loss)/3

        if not is_training:
            return

        reg_penalty = encoding_l2 + pos_l2 + chunk_l2 + lm_l2 + input_l2

        self.pos_op = _training(pos_loss + pos_l2, config, self)
        self.chunk_op = _training(chunk_loss + chunk_l2, config, self)
        self.lm_op = _training((1-config.reg_weight)*(lm_loss) + config.reg_weight*reg_penalty, config, self)
        self.joint_op = _training((1-config.reg_weight)*(chunk_loss + pos_loss + lm_loss )/3 + \
                                    config.reg_weight*reg_penalty, config, self)
