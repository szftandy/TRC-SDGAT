import logging
import tensorflow as tf
import numpy as np
from math import log
from .encoder import TransformerEncoder, GATEncoder, SDGATEncoder
from .decoder import TransformerDecoder, SingleLayerDecoder, BilinearDecoder, NTNDecoder
from .commons import Replayer, Critic
from .commons import variable_summaries

class PPO_RB_Agent(object):
    _logger = logging.getLogger(__name__)
    def __init__(self, config):
        self.config = config
        self.is_train = True
        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  
        self.input_dimension = config.input_dimension  

        # Training config (actor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step

        # Training config (critic)
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2")  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_coordinates")
        self.reward_ = tf.placeholder(tf.float32, [self.batch_size], name='input_rewards')
        self.graphs_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.max_length], name='input_graphs')
        self.log_pi_old_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.max_length], name='old_policy')
        self.advantage_ = tf.placeholder(tf.float32, [self.batch_size], name='old_policy')
        self.clip_ratio = 0.1
        self.rb_rate = 0.3
        
        # self.replayer = Replayer(config, prioritized=True, size_times=10)
        self.replayer = Replayer(config, prioritized=False, size_times=1)
        self.build_actor()
        self.build_critic()
        self.build_advantage()
        self.build_optimization()
        self.merged = tf.summary.merge_all()

    def build_actor(self):
        with tf.variable_scope("encoder"):
            if self.config.encoder_type == 'TransformerEncoder':
                encoder = TransformerEncoder(self.config, self.is_train)
            elif self.config.encoder_type == 'GATEncoder':
                encoder = GATEncoder(self.config, self.is_train)
            elif self.config.encoder_type == 'SDGATEncoder':
                encoder = SDGATEncoder(self.config, self.is_train)
            else:
                raise NotImplementedError('Current encoder type is not implemented yet!')
            self.encoder_output = encoder.encode(self.input_)

        with tf.variable_scope('decoder'):
            if self.config.decoder_type == 'SingleLayerDecoder':
                self.decoder = SingleLayerDecoder(self.config, self.is_train)
            elif self.config.decoder_type == 'TransformerDecoder':
                self.decoder = TransformerDecoder(self.config, self.is_train)
            elif self.config.decoder_type == 'BilinearDecoder':
                self.decoder = BilinearDecoder(self.config, self.is_train)
            elif self.config.decoder_type == 'NTNDecoder':
                self.decoder = NTNDecoder(self.config, self.is_train)
            else:
                raise NotImplementedError('Current decoder type is not implemented yet!')

            self.samples, self.scores, self.entropy = self.decoder.decode(self.encoder_output)
            
            self.graphs = tf.transpose(tf.stack(self.samples), [1, 0, 2])
            gen_graphs = tf.cast(self.graphs, tf.float32)
            self.logits_for_rewards = tf.transpose(tf.stack(self.scores), [1, 0, 2])
            entropy_for_rewards = tf.transpose(tf.stack(self.entropy), [1, 0, 2])
            self.entropy_regularization = tf.reduce_mean(entropy_for_rewards, axis=[1,2])
            self.graph_batch = tf.reduce_mean(self.graphs, axis=0)
            variable_summaries('entropy', self.entropy_regularization, with_max_min=True)
        
        with tf.name_scope('pi'):
            # Calculate pi(a)
            self.log_pi = -tf.nn.sigmoid_cross_entropy_with_logits(labels=gen_graphs, logits=self.logits_for_rewards)
            self.log_pi_new = -tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.graphs_, tf.float32), logits=self.logits_for_rewards)
            variable_summaries('log_pi', self.log_pi, with_max_min=True)
            
    def build_critic(self):
        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config, self.is_train)
            self.critic.predict(self.encoder_output)
            variable_summaries('predictions', self.critic.predictions, with_max_min=True)

    def build_advantage(self):
        with tf.name_scope('advantage'):
            self.advantage = self.advantage_
            variable_summaries('advantage', self.advantage, with_max_min=True)

    def build_optimization(self):
        # Update moving_mean and moving_variance for batch normalization layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('surrogate_update'):
                # Actor learning rate
                self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,
                                                      self.lr1_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1, beta1=0.9, beta2=0.99, epsilon=0.0000001)         
                # rollback
                self.log_pi_old = self.log_pi_old_
                self.log_gap = self.log_pi_new - self.log_pi_old
                self.log_gap_clip = tf.clip_by_value(self.log_gap, log(1-self.clip_ratio), log(1+self.clip_ratio))
                self.ratio_1 = tf.exp(tf.reduce_sum(self.log_gap, axis=[1,2]))
                self.ratio_2 = tf.exp(tf.reduce_sum(self.log_gap_clip, axis=[1,2]))
                self.ratio_rb = ((1+self.rb_rate)*self.ratio_2-self.rb_rate*self.ratio_1)
                self.surr1 =  self.ratio_1 * self.advantage
                self.surr2 =  self.ratio_rb * self.advantage
                self.clipped_surrogate = -tf.reduce_mean(tf.minimum(self.surr1, self.surr2))
                self.loss1 = self.clipped_surrogate - self.lr1 * tf.reduce_mean(self.entropy_regularization, 0)

                tf.summary.scalar('loss1', self.loss1)
                # Minimize step
                gvs = self.opt1.compute_gradients(self.loss1)
                gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip
                self.train_step1 = self.opt1.apply_gradients(gvs, global_step=self.global_step)

            with tf.name_scope('state_value'):
                # Critic learning rate
                self.lr2 = tf.train.exponential_decay(self.lr2_start, self.global_step2, self.lr2_decay_step,
                                                      self.lr2_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                # Loss
                weights_ = 1.0  # weights_ = tf.exp(self.log_softmax-tf.reduce_max(self.log_softmax)) # probs / max_prob
                self.reward = self.reward_
                self.loss2 = tf.losses.mean_squared_error(self.reward, self.critic.predictions, weights=weights_)
                tf.summary.scalar('loss2', self.loss2)
                # Minimize step
                gvs2 = self.opt2.compute_gradients(self.loss2)
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None]  # L2 clip
                self.train_step2 = self.opt1.apply_gradients(capped_gvs2, global_step=self.global_step2)
