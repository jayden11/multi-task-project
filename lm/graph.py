from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

import pdb

class Shared_Model(object):

    def __init__(self, config, is_training, num_pos_tags, num_chunk_tags,
        vocab_size, word_embedding, projection_size):
        """Initialisation
            basically set the self-variables up, so that we can call them
            as variables to the model.
        """
        self.max_grad_norm = max_grad_norm = config.max_grad_norm
        self.num_steps = num_steps = config.num_steps
        self.encoder_size = encoder_size = config.encoder_size
        self.pos_decoder_size = pos_decoder_size = config.pos_decoder_size
        self.chunk_decoder_size = chunk_decoder_size = config.chunk_decoder_size
        self.batch_size = batch_size = config.batch_size
        self.vocab_size = vocab_size
        self.num_pos_tags = num_pos_tags
        self.num_chunk_tags = num_chunk_tags
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.word_embedding_size = word_embedding_size = config.word_embedding_size
        self.pos_embedding_size = pos_embedding_size = config.pos_embedding_size
        self.chunk_embedding_size = chunk_embedding_size = config.chunk_embedding_size
        self.num_shared_layers = num_shared_layers = config.num_shared_layers
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

        self.gold_embed = tf.placeholder(tf.int32, shape=[], name="condition")

        def _shared_layer(input_data, config):
            """Build the model to decoding

            Args:
                input_data = size batch_size X num_steps X embedding size

            Returns:
                output units
            """

            if config.bidirectional == True:
                if config.lstm == True:
                    cell_fw = rnn_cell.BasicLSTMCell(config.encoder_size, forget_bias = 1.0)
                    cell_bw = rnn_cell.BasicLSTMCell(config.encoder_size, forget_bias = 1.0)
                else:
                    cell_fw = rnn_cell.GRUCell(config.encoder_size)
                    cell_bw = rnn_cell.GRUCell(config.encoder_size)

                inputs = [tf.squeeze(input_, [1])
                          for input_ in tf.split(1, config.num_steps, input_data)]

                if is_training and config.keep_prob < 1:
                    cell_fw = rnn_cell.DropoutWrapper(
                        cell_fw, output_keep_prob=config.keep_prob)
                    cell_bw = rnn_cell.DropoutWrapper(
                        cell_bw, output_keep_prob=config.keep_prob)


                cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
                cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

                initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
                initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

                encoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      scope="encoder_rnn")


            else:
                if config.lstm == True:
                    cell = rnn_cell.BasicLSTMCell(config.encoder_size)
                else:
                    cell = rnn_cell.GRUCell(config.encoder_size)

                inputs = [tf.squeeze(input_, [1])
                          for input_ in tf.split(1, config.num_steps, input_data)]

                if is_training and config.keep_prob < 1:
                    cell = rnn_cell.DropoutWrapper(
                        cell, output_keep_prob=config.keep_prob)

                cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

                initial_state = cell.zero_state(config.batch_size, tf.float32)

                encoder_outputs, encoder_states = rnn.rnn(cell, inputs,
                                                          initial_state=initial_state,
                                                          scope="encoder_rnn")

            return encoder_outputs

        def _pos_private(encoder_units, config):
            """Decode model for pos

            Args:
                encoder_units - these are the encoder units
                num_pos - the number of pos tags there are (output units)

            returns:
                logits
            """
            with tf.variable_scope("pos_decoder"):
                if config.bidirectional == True:
                    if config.lstm == True:
                        cell_fw = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                                      forget_bias=1.0)
                        cell_bw = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                                      forget_bias=1.0)
                    else:
                        cell_fw = rnn_cell.GRUCell(config.pos_decoder_size)
                        cell_bw = rnn_cell.GRUCell(config.pos_decoder_size)

                    if is_training and config.keep_prob < 1:
                        cell_fw = rnn_cell.DropoutWrapper(
                            cell_fw, output_keep_prob=config.keep_prob)
                        cell_bw = rnn_cell.DropoutWrapper(
                            cell_bw, output_keep_prob=config.keep_prob)

                    cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
                    cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

                    initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
                    initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

                    # puts it into batch_size X input_size
                    inputs = [tf.squeeze(input_, [1])
                              for input_ in tf.split(1, config.num_steps,
                                                     encoder_units)]

                    decoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                                              initial_state_fw=initial_state_fw,
                                                              initial_state_bw=initial_state_bw,
                                                              scope="pos_rnn")

                    output = tf.reshape(tf.concat(1, decoder_outputs),
                                        [-1, 2*config.pos_decoder_size])

                    softmax_w = tf.get_variable("softmax_w",
                                                [2*config.pos_decoder_size,
                                                 num_pos_tags])
                else:
                    if config.lstm == True:
                        cell = rnn_cell.BasicLSTMCell(config.pos_decoder_size,
                                                  forget_bias=1.0)
                    else:
                        cell = rnn_cell.GRUCell(config.pos_decoder_size)

                    if is_training and config.keep_prob < 1:
                        cell = rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=config.keep_prob)

                    cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

                    initial_state = cell.zero_state(config.batch_size, tf.float32)

                    # puts it into batch_size X input_size
                    inputs = [tf.squeeze(input_, [1])
                              for input_ in tf.split(1, config.num_steps,
                                                     encoder_units)]

                    decoder_outputs, decoder_states = rnn.rnn(cell, inputs,
                                                              initial_state=initial_state,
                                                              scope="pos_rnn")
                    output = tf.reshape(tf.concat(1, decoder_outputs),
                                        [-1, config.pos_decoder_size])

                    softmax_w = tf.get_variable("softmax_w",
                                                [config.pos_decoder_size,
                                                 num_pos_tags])

                softmax_b = tf.get_variable("softmax_b", [num_pos_tags])
                logits = tf.matmul(output, softmax_w) + softmax_b
                l2_penalty = tf.reduce_sum(tf.square(output))

            return logits, l2_penalty

        def _chunk_private(encoder_units, pos_prediction, config):
            """Decode model for chunks

            Args:
                encoder_units - these are the encoder units:
                [batch_size X encoder_size] with the one the pos prediction
                pos_prediction:
                must be the same size as the encoder_size

            returns:
                logits
            """
            # concatenate the encoder_units and the pos_prediction

            pos_prediction = tf.reshape(pos_prediction,
                [batch_size, num_steps, pos_embedding_size])
            chunk_inputs = tf.concat(2, [pos_prediction, encoder_units])

            with tf.variable_scope("chunk_decoder"):
                if config.bidirectional == True:
                    if config.lstm == True:
                        cell_fw = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)
                        cell_bw = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)
                    else:
                        cell_fw = rnn_cell.GRUCell(config.chunk_decoder_size)
                        cell_bw = rnn_cell.GRUCell(config.chunk_decoder_size)

                    if is_training and config.keep_prob < 1:
                        cell_fw = rnn_cell.DropoutWrapper(
                            cell_fw, output_keep_prob=config.keep_prob)
                        cell_bw = rnn_cell.DropoutWrapper(
                            cell_bw, output_keep_prob=config.keep_prob)

                    cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
                    cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

                    initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
                    initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

                    # this function puts the 3d tensor into a 2d tensor: batch_size x input size
                    inputs = [tf.squeeze(input_, [1])
                              for input_ in tf.split(1, config.num_steps,
                                                     chunk_inputs)]

                    decoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw,
                                                              inputs, initial_state_fw=initial_state_fw,
                                                              initial_state_bw=initial_state_bw,
                                                              scope="chunk_rnn")
                    output = tf.reshape(tf.concat(1, decoder_outputs),
                                        [-1, 2*config.chunk_decoder_size])
                    softmax_w = tf.get_variable("softmax_w",
                                                [2*config.chunk_decoder_size,
                                                 num_chunk_tags])
                else:
                    if config.lstm == True:
                        cell = rnn_cell.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0)
                    else:
                        cell = rnn_cell.GRUCell(config.chunk_decoder_size)

                    if is_training and config.keep_prob < 1:
                        cell = rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=config.keep_prob)

                    cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

                    initial_state = cell.zero_state(config.batch_size, tf.float32)

                    # this function puts the 3d tensor into a 2d tensor: batch_size x input size
                    inputs = [tf.squeeze(input_, [1])
                              for input_ in tf.split(1, config.num_steps,
                                                     chunk_inputs)]

                    decoder_outputs, decoder_states = rnn.rnn(cell,
                                                              inputs, initial_state=initial_state,
                                                              scope="chunk_rnn")

                    output = tf.reshape(tf.concat(1, decoder_outputs),
                                        [-1, config.chunk_decoder_size])

                    softmax_w = tf.get_variable("softmax_w",
                                                [config.chunk_decoder_size,
                                                 num_chunk_tags])

                softmax_b = tf.get_variable("softmax_b", [num_chunk_tags])
                logits = tf.matmul(output, softmax_w) + softmax_b
                l2_penalty = tf.reduce_sum(tf.square(output))

            return logits, l2_penalty

        def _lm_private(encoder_units, pos_prediction, chunk_prediction, config):
            """Decode model for lm

            Args:
                encoder_units - these are the encoder units:
                [batch_size X encoder_size] with the one the pos prediction
                pos_prediction:
                must be the same size as the encoder_size

            returns:
                logits
            """
            # concatenate the encoder_units and the pos_prediction

            pos_prediction = tf.reshape(pos_prediction,
                [batch_size, num_steps, pos_embedding_size])
            chunk_prediction = tf.reshape(chunk_prediction,
                [batch_size, num_steps, chunk_embedding_size])
            lm_inputs = tf.concat(2, [chunk_prediction, pos_prediction, encoder_units])

            with tf.variable_scope("lm_decoder"):
                if config.bidirectional == True:
                    if config.lstm == True:
                        cell_fw = rnn_cell.BasicLSTMCell(config.lm_decoder_size, forget_bias=1.0)
                        cell_bw = rnn_cell.BasicLSTMCell(config.lm_decoder_size, forget_bias=1.0)
                    else:
                        cell_fw = rnn_cell.GRUCell(config.lm_decoder_size)
                        cell_bw = rnn_cell.GRUCell(config.lm_decoder_size)

                    if is_training and config.keep_prob < 1:
                        cell_fw = rnn_cell.DropoutWrapper(
                            cell_fw, output_keep_prob=config.keep_prob)
                        cell_bw = rnn_cell.DropoutWrapper(
                            cell_bw, output_keep_prob=config.keep_prob)

                    cell_fw = rnn_cell.MultiRNNCell([cell_fw] * config.num_shared_layers)
                    cell_bw = rnn_cell.MultiRNNCell([cell_bw] * config.num_shared_layers)

                    initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
                    initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

                    # this function puts the 3d tensor into a 2d tensor: batch_size x input size
                    inputs = [tf.squeeze(input_, [1])
                              for input_ in tf.split(1, config.num_steps,
                                                     lm_inputs)]

                    decoder_outputs, _, _ = rnn.bidirectional_rnn(cell_fw, cell_bw,
                                                              inputs, initial_state_fw=initial_state_fw,
                                                              initial_state_bw=initial_state_bw,
                                                              scope="lm_rnn")
                    output = tf.reshape(tf.concat(1, decoder_outputs),
                                        [-1, 2*config.lm_decoder_size])
                    softmax_w = tf.get_variable("softmax_w",
                                                [2*config.lm_decoder_size,
                                                 vocab_size])
                else:
                    if config.lstm == True:
                        cell = rnn_cell.BasicLSTMCell(config.lm_decoder_size)
                    else:
                        cell = rnn_cell.GRUCell(config.lm_decoder_size)

                    if is_training and config.keep_prob < 1:
                        cell = rnn_cell.DropoutWrapper(
                            cell, output_keep_prob=config.keep_prob)

                    cell = rnn_cell.MultiRNNCell([cell] * config.num_shared_layers)

                    initial_state = cell.zero_state(config.batch_size, tf.float32)

                    # this function puts the 3d tensor into a 2d tensor: batch_size x input size
                    inputs = [tf.squeeze(input_, [1])
                              for input_ in tf.split(1, config.num_steps,
                                                     lm_inputs)]

                    decoder_outputs, decoder_states = rnn.rnn(cell,
                                                              inputs, initial_state=initial_state,
                                                              scope="lm_rnn")

                    output = tf.reshape(tf.concat(1, decoder_outputs),
                                        [-1, config.lm_decoder_size])
                    softmax_w = tf.get_variable("softmax_w",
                                                [config.lm_decoder_size,
                                                 vocab_size])

                softmax_b = tf.get_variable("softmax_b", [vocab_size])
                logits = tf.matmul(output, softmax_w) + softmax_b
                l2_penalty = tf.reduce_sum(tf.square(output))

            return logits, l2_penalty

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


        #################################################################
        # Section 2: Construct the graph from the functions defined above
        # ==============================================================
        # Awesome sauce
        ################################################################

        # Read in the embeddings in a memory efficient manner
        word_embedding = word_embedding = tf.get_variable("word_embedding", [vocab_size, word_embedding_size], trainable=False)
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_embedding_size])
        self.embedding_init = word_embedding.assign(self.embedding_placeholder)

        # get the embeddings
        inputs = tf.nn.embedding_lookup(word_embedding, self.input_data)
        inputs = input_projection3D(inputs, projection_size) # put them through a projections
        input_l2 = tf.reduce_sum(inputs) # sum them up for the regulariser

        # get the pos and chunk embeddings
        self.pos_embedding = pos_embedding = tf.get_variable("pos_embedding",
            [num_pos_tags, pos_embedding_size])
        self.chunk_embedding = chunk_embedding = tf.get_variable("chunk_embedding",
            [num_chunk_tags, chunk_embedding_size])

        # add dropout if training
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # create the shared layer (called the encoding)
        encoding = _shared_layer(inputs, config)
        encoding = tf.pack(encoding)
        encoding = tf.transpose(encoding, perm=[1, 0, 2])
        encoding_l2 = tf.reduce_sum(encoding) # for the regulariser

        # create the pos layer
        pos_logits, pos_l2 = _pos_private(encoding, config)
        pos_loss, pos_accuracy, pos_int_pred, pos_int_targ = _loss(pos_logits, self.pos_targets)

        # expose these values to the world
        self.pos_loss = pos_loss
        self.pos_int_pred = pos_int_pred
        self.pos_int_targ = pos_int_targ

        # choose either argmax or dot product for pos embedding
        if config.argmax==1:
            pos_to_chunk_embed = tf.cond(self.gold_embed > 0 , lambda: tf.matmul(self.pos_targets, pos_embedding),\
            lambda: tf.nn.embedding_lookup(pos_embedding,pos_int_pred))
        else:
            pos_to_chunk_embed = tf.cond(self.gold_embed > 0 , lambda: tf.matmul(self.pos_targets, pos_embedding), \
            lambda: tf.matmul(tf.nn.softmax(pos_logits),pos_embedding))

        # create the chunk layer
        chunk_logits, chunk_l2 = _chunk_private(encoding, pos_to_chunk_embed, config)
        chunk_loss, chunk_accuracy, chunk_int_pred, chunk_int_targ = _loss(chunk_logits, self.chunk_targets)
        # expose the values to the world
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

        # create the LM layer
        lm_logits, lm_l2 = _lm_private(encoding, chunk_to_lm_embed,  pos_to_chunk_embed, config)
        lm_loss, lm_accuracy, lm_int_pred, lm_int_targ = _loss(lm_logits, self.lm_targets)
        # expose these values to the world
        self.lm_loss = lm_loss
        self.lm_int_pred = lm_int_pred
        self.lm_int_targ = lm_int_targ

        # define the joint loss
        self.joint_loss = (chunk_loss + pos_loss + lm_loss)/3
        if not is_training:
            return

        # define the regulariser parameters
        total_l2 = lm_l2 + pos_l2 + chunk_l2 + encoding_l2 + input_l2

        self.pos_op = _training(pos_loss, config, self)
        self.chunk_op = _training(chunk_loss, config, self)
        self.lm_op = _training(lm_loss, config, self)
        self.joint_op = _training((chunk_loss + pos_loss + lm_loss)/3 + config.reg_weight*total_l2, config, self)
