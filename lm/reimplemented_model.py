import tensorflow as tf
import pdb

class ReimplementedModel:
    def __init__(self, config, num_pos_tags, num_chunk_tags,
    vocab_size,  word_embedding, is_training):
        self.num_steps = num_steps = config.num_steps
        self.layer_size = layer_size =config.layer_size
        self.batch_size = batch_size = config.batch_size
        self.vocab_size = vocab_size
        self.num_pos_tags = num_pos_tags
        self.num_chunk_tags = num_chunk_tags
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.word_embedding_size = word_embedding_size = config.word_embedding_size
        self.connection_embedding_size = connection_embedding_size = config.connection_embedding_size
        self.mix_percent = mix_percent = config.mix_percent
        self.pos_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                          num_pos_tags])
        self.chunk_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                            num_chunk_tags])
        self.lm_targets = tf.placeholder(tf.float32, [(batch_size*num_steps),
                                            vocab_size])
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_embedding_size])
        word_embedding = tf.get_variable("word_embedding",
            [vocab_size, word_embedding_size])
        self.embedding_init = word_embedding.assign(self.embedding_placeholder)

        def _loss(logits, labels):
            """calculate the loss
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
            """Defines the training loss
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
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
            return train_op

        def _repack(layer_outputs):
            return [tf.squeeze(input_, [1])
                      for input_ in tf.split(1, config.num_steps,
                                             layer_outputs)]

        pos_embedding = tf.get_variable("pos_embedding",
            [num_pos_tags, connection_embedding_size])
        chunk_embedding = tf.get_variable("chunk_embedding",
            [num_chunk_tags, connection_embedding_size])

        # shared layer
        embedded_inputs = tf.nn.embedding_lookup(word_embedding, self.input_data)

        if is_training and config.keep_prob < 1:
            embedded_inputs = tf.nn.dropout(embedded_inputs, config.keep_prob)

        embedded_inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, config.num_steps, embedded_inputs)]

        cell_fw = tf.nn.rnn_cell.GRUCell(config.layer_size)
        cell_bw = tf.nn.rnn_cell.GRUCell(config.layer_size)

        if is_training and config.keep_prob < 1:
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                cell_fw, output_keep_prob=config.keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                cell_bw, output_keep_prob=config.keep_prob)

        cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw])
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw])

        initial_state_fw = cell_fw.zero_state(config.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(config.batch_size, tf.float32)

        shared_layer_outputs, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, embedded_inputs,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw,
                                              scope="shared_rnn")

        pos_cell_fw = tf.nn.rnn_cell.GRUCell(config.layer_size)
        pos_cell_bw = tf.nn.rnn_cell.GRUCell(config.layer_size)

        if is_training and config.keep_prob < 1:
            pos_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                pos_cell_fw, output_keep_prob=config.keep_prob)
            pos_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                pos_cell_bw, output_keep_prob=config.keep_prob)

        pos_cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw])
        pos_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw])

        pos_layer_hidden_outputs, _, _ = tf.nn.bidirectional_rnn(pos_cell_fw, pos_cell_bw, shared_layer_outputs,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw,
                                              scope="pos_rnn")

        pos_layer_hidden_outputs = tf.reshape(tf.concat(1, pos_layer_hidden_outputs),
                            [-1, 2*config.layer_size])

        pos_softmax_w = tf.get_variable("pos_softmax_w",
                                    [2*config.layer_size,
                                     num_pos_tags])

        pos_softmax_b = tf.get_variable("pos_softmax_b", [num_pos_tags])
        pos_layer_logits = tf.matmul(pos_layer_hidden_outputs, pos_softmax_w) + pos_softmax_b

        pos_embedded_layer = tf.matmul(tf.nn.softmax(pos_layer_logits),pos_embedding)
        pos_embedded_layer = tf.reshape(pos_embedded_layer,
            [config.batch_size, config.num_steps, config.connection_embedding_size])
        pos_embedded_layer = tf.unpack(tf.transpose(pos_embedded_layer, perm=[1, 0, 2]))

        chunk_inputs = [tf.concat(1,[a,b]) for a,b in zip(pos_embedded_layer,shared_layer_outputs)]

        # chunk layer
        chunk_cell_fw = tf.nn.rnn_cell.GRUCell(config.layer_size)
        chunk_cell_bw = tf.nn.rnn_cell.GRUCell(config.layer_size)

        if is_training and config.keep_prob < 1:
            chunk_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                chunk_cell_fw, output_keep_prob=config.keep_prob)
            chunk_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                chunk_cell_bw, output_keep_prob=config.keep_prob)

        chunk_cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw])
        chunk_cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw])

        chunk_layer_hidden_outputs, _, _ = tf.nn.bidirectional_rnn(chunk_cell_fw, chunk_cell_bw, chunk_inputs,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw,
                                              scope="chunk_rnn")

        chunk_layer_hidden_outputs = tf.reshape(tf.concat(1, chunk_layer_hidden_outputs),
                            [-1, 2*config.layer_size])

        chunk_softmax_w = tf.get_variable("chunk_softmax_w",
                                    [2*config.layer_size,
                                     num_chunk_tags])

        chunk_softmax_b = tf.get_variable("chunk_softmax_b", [num_chunk_tags])
        chunk_layer_logits = tf.matmul(chunk_layer_hidden_outputs, chunk_softmax_w) + chunk_softmax_b

        chunk_embedded_layer = tf.matmul(tf.nn.softmax(chunk_layer_logits),chunk_embedding)
        chunk_embedded_layer = tf.reshape(chunk_embedded_layer,
            [config.batch_size, config.num_steps, config.connection_embedding_size])
        chunk_embedded_layer = tf.unpack(tf.transpose(chunk_embedded_layer, perm=[1, 0, 2]))

        lm_inputs = [tf.concat(1, [a,b,c]) for a,b,c in zip(pos_embedded_layer, chunk_embedded_layer, shared_layer_outputs)]

        # lm layer
        lm_cell_fw = tf.nn.rnn_cell.GRUCell(config.layer_size)
        lm_cell_bw = tf.nn.rnn_cell.GRUCell(config.layer_size)

        lm_layer_hidden_outputs, _, _ = tf.nn.bidirectional_rnn(lm_cell_fw, lm_cell_bw, lm_inputs,
                                              initial_state_fw=initial_state_fw,
                                              initial_state_bw=initial_state_bw,
                                              scope="lm_rnn")

        lm_layer_hidden_outputs = tf.reshape(tf.concat(1, lm_layer_hidden_outputs),
                    [-1, 2*config.layer_size])

        lm_softmax_w = tf.get_variable("lm_softmax_w",
                                    [2*config.layer_size,
                                     vocab_size])

        lm_softmax_b = tf.get_variable("lm_softmax_b", [vocab_size])
        lm_layer_logits = tf.matmul(lm_layer_hidden_outputs, lm_softmax_w) + lm_softmax_b

        pos_loss, pos_accuracy, pos_int_pred, pos_int_targ = _loss(pos_layer_logits, self.pos_targets)
        chunk_loss, chunk_accuracy, chunk_int_pred, chunk_int_targ = _loss(chunk_layer_logits, self.chunk_targets)
        lm_loss, lm_accuracy, lm_int_pred, lm_int_targ = _loss(lm_layer_logits, self.lm_targets)

        self.pos_op = _training(pos_loss, config, self)
        self.chunk_op = _training(chunk_loss, config, self)
        self.lm_op = _training(lm_loss, config, self)
        self.joint_op = _training((chunk_loss + pos_loss + lm_loss)/3,
                                    config, self)
        self.chunk_and_pos_op = _training(pos_loss + chunk_loss/2, config, self)

        self.pos_loss = pos_loss
        self.chunk_loss = chunk_loss
        self.lm_loss = lm_loss
        self.pos_int_pred = pos_int_pred
        self.pos_int_targ = pos_int_targ
        self.chunk_int_pred = chunk_int_pred
        self.chunk_int_targ = chunk_int_targ
        self.lm_int_pred = lm_int_pred
        self.lm_int_targ = lm_int_targ
        self.joint_loss = pos_loss + chunk_loss + lm_loss
