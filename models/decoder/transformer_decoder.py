import tensorflow as tf
from tensorflow.contrib import distributions as distr

def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    with tf.variable_scope("multihead_attention", reuse=None):
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]
        outputs += inputs # [batch_size, seq_length, n_hidden]
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
    return outputs

def feedforward(inputs, num_units=[2048, 512], is_training=True):
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs += inputs
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
    return outputs

class TransformerDecoder(object):
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim#input_dimension*2+1 # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 high priority token, 1 pointing
        self.input_embed = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.max_length = config.max_length
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.is_training = is_train
        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

    def decode(self, inputs):
        all_user_embedding = tf.reduce_mean(inputs, 1)
        inputs_with_all_user_embedding = tf.concat([inputs, tf.tile(tf.expand_dims(all_user_embedding,1), [1, self.max_length ,1])], -1)
        with tf.variable_scope("embedding_MCS"):
            # Embed input sequence
            W_embed =tf.get_variable("weights",[1, self.input_embed , self.input_embed], initializer=self.initializer) #self.input_dimension*2
            self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            # Batch Normalization
            self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        with tf.variable_scope("stack_MCS"):
            for i in range(self.num_stacks): # num blocks
                with tf.variable_scope("block_{}".format(i)):
                    self.enc = multihead_attention(self.enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
                    self.enc = feedforward(self.enc, num_units=[self.input_embed, self.input_embed], is_training=self.is_training)
            params = {"inputs": self.enc, "filters": self.max_length, "kernel_size": 1, "activation": None, "use_bias": True}
            self.adj_prob = tf.layers.conv1d(**params)
            for i in range(self.max_length):
                position = tf.ones([inputs.shape[0]]) * i
                position = tf.cast(position, tf.int32)
                self.mask = tf.one_hot(position, self.max_length)
                masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
                prob = distr.Bernoulli(masked_score)#probs input probability, logit input log_probability
                sampled_arr = prob.sample() # Batch_size, seqlenght for just one node
                self.samples.append(sampled_arr)
                self.mask_scores.append(masked_score)
                self.entropy.append(prob.entropy())
        return self.samples, self.mask_scores, self.entropy
 