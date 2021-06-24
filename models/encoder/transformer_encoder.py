import tensorflow as tf
 
class TransformerEncoder(object):
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.input_embed = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.is_training = is_train #not config.inference_mode
 
    def encode(self, inputs):
        with tf.variable_scope("embedding"):
            W_embed =tf.get_variable("weights",[1, self.input_dimension, self.input_embed], initializer=self.initializer)
            self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        with tf.variable_scope("stack"):
            for i in range(self.num_stacks): # num blocks
                with tf.variable_scope("block_{}".format(i)):
                    self.enc = self.multihead_attention(self.enc)
                    self.enc = self.feedforward(self.enc, num_units=[4*self.input_embed, self.input_embed])
            self.encoder_output = self.enc
        return self.encoder_output
 
    def multihead_attention(self, inputs, dropout_rate=0.1):
        with tf.variable_scope("multihead_attention", reuse=None):
            # Linear projections
            Q = tf.layers.dense(inputs, self.input_embed, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
            K = tf.layers.dense(inputs, self.input_embed, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
            V = tf.layers.dense(inputs, self.input_embed, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
            # Split and concat
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(self.is_training))
            outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]
            outputs += inputs # [batch_size, seq_length, n_hidden]
            outputs = tf.layers.batch_normalization(outputs, axis=2, training=self.is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
        return outputs
     
    def feedforward(self, inputs, num_units=[2048, 512]):
        with tf.variable_scope("ffn", reuse=None):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs += inputs
            outputs = tf.layers.batch_normalization(outputs, axis=2, training=self.is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
        return outputs
 
 
 
