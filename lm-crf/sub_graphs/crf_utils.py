import tensorflow
import numpy
import pdb

def evaluate_score(hidden_states, A, W, y):
    # hidden_states - list of batch_size X num_tags tensors
    # A - transition matrix
    # W - token matrix
    # y - sequences - batch_size X num_steps

    # initialise the variables
    unary_score = 0
    transition_score = 0

    # split y into batch sizes
    split_y = tf.split(y, axis=1)

    # calculate unary score
    for h_t, y_t in zip(hidden_states, split_y):
        w_t = tf.nn.embedding_lookup(W,y_t)
        unary_score += tf.matmul(tf.transpose(w_t), h_t)

    # unroll y
    unrolled_y = tf.squeeze(y)

    # calculate transition score
    t_prev = 0
    for y_next in unrolled_y:
        transition_score += A[y_prev,y_next]

    return transition_score + unary_score

def sum_product(hidden_states, A, W):

    def transition_cost_vector(A,t_cost):
        return tf.matmul(A,t_cost)

    def unary_cost_vector(W,h):
        return tf.matmul(tf.transpose(W),h)

    t_cost = tf.ones(w.get_shape()[0], float32)
    norm_constant = tf.ones(w.get_shape()[0], float32)

    for h in hidden_states:
        t_cost = transition_cost_vector(A,t_cost)
        u_cost = unary_cost_vector(W,h)
        norm_constant = tf.mul(tf.exp(t_cost+u_cost),norm_constant)

    norm_constant = tf.log(tf.reduce_sum(norm_constant))

    return norm_constant

def max_product(hidden_states, A, W):

    def transition_vector(A,t_cost):
        return tf.reduce_max(A,t_cost)

    def unary_vector(W,h):
        return tf.matmul(tf.transpose(W),h)

    vocab_size = w.get_shape()[0]
    t_vector = tf.ones(vocab_size, float32)
    norm_constant = tf.ones(vocab_size, float32)

    for h in hidden_states:
        t_cost = transition_vector(A,t_cost)
        u_cost = unary_vector(W,h)
        max_val, max_idx = tf.top_k(tf.exp(t_cost+u_cost))
        t_cost = tf.one_hot(max_idx, vocab_size)
        norm_constant = tf.mul(max_val,norm_constant)

    norm_constant = tf.log(tf.reduce_sum(norm_constant))

    return norm_constant
