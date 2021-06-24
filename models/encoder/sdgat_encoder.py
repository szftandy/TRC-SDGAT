import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def multihead_attention(inputs, num_units=None, num_heads=4, dropout_rate=0.0, is_training=True):
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
    # outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=True)  # [batch_size, seq_length, n_hidden]
    return outputs

class SDGATEncoder(object):
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.hidden_dim = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.gat_stacks = config.num_gat_stacks
        self.residual = config.residual
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.is_training = is_train #not config.inference_mode

    def encode(self, inputs):
        """
        input shape: (batch_size, max_length, input_dimension)
        output shape: (batch_size, max_length, input_embed)
        """
        with tf.variable_scope("embedding"):
            W_embed = tf.get_variable("weights",[1, self.input_dimension, self.hidden_dim], initializer=self.initializer)
            embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            self.encs = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        
        head_hidden_dim = self.hidden_dim / self.num_heads
        attns = []
        h_1 = tf.split(self.encs, self.num_heads, axis=-1)
        for i in range(self.num_heads):
            attns.append(self.attn_head(h_1[i], out_sz=head_hidden_dim, num_units=self.hidden_dim, activation=tf.nn.elu, in_drop=0, coef_drop=0))
        
        for _ in range(self.gat_stacks-2):
            for i in range(self.num_heads):
                attns[i] = self.attn_head(attns[i], out_sz=head_hidden_dim, num_units=self.hidden_dim, activation=tf.nn.elu, in_drop=0, coef_drop=0)
        h_1 = tf.concat(attns, axis=-1)
        
        out = []
        for i in range(self.num_heads):
            out.append(self.attn_head(attns[i], out_sz=self.hidden_dim, num_units=self.hidden_dim, activation=tf.nn.elu, in_drop=0, coef_drop=0))
        out = tf.add_n(out) / self.num_heads
        return out
        
    def attn_head(self, seq, out_sz, num_units, activation, in_drop=0.0, coef_drop=0.0):
        with tf.name_scope('sdgat_attn'):
            if in_drop != 0.0:
                seq = tf.nn.dropout(seq, 1.0 - in_drop)
            vals = multihead_attention(seq, out_sz)
            ret = tf.layers.dense(vals, out_sz, activation=tf.nn.relu)
            if self.residual:
                if seq.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1)
                else:
                    ret = ret + seq
            return activation(ret)
        # attns = []
        # for _ in range(n_heads[0]):
        #     attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
        #         out_sz=hid_units[0], activation=activation,
        #         in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        # h_1 = tf.concat(attns, axis=-1)
        # for i in range(1, len(hid_units)):
        #     h_old = h_1
        #     attns = []
        #     for _ in range(n_heads[i]):
        #         attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #             out_sz=hid_units[i], activation=activation,
        #             in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
        #     h_1 = tf.concat(attns, axis=-1)
        # out = []
        # for i in range(n_heads[-1]):
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #         out_sz=nb_classes, activation=lambda x: x,
        #         in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        # logits = tf.add_n(out) / n_heads[-1]
    
        # return logits
    

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    from utils.config_utils import get_config
    config, _ = get_config()
    inputs = tf.placeholder(tf.float32, [config.batch_size, config.max_length, config.input_dimension])
    net = SDGATEncoder(config, is_train=True)
    out = net.encode(inputs)
    