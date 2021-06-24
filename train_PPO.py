import os
import logging
import platform
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from models import PPO, PPO_RB, TRC
from environment import Reward_BIC as reward
from utils.data_loader import DataGenerator
from utils.config_utils import get_config
from utils.dir_utils import create_dir
from utils.log_utils import LogHelper
from utils.tf_utils import set_seed
from utils.analyze_utils import convert_graph_int_to_adj_mat, graph_prunned_by_coef, count_accuracy, graph_prunned_by_coef_2nd
from utils.cam_with_pruning_cam import pruning_cam
from utils.lambda_utils import BIC_lambdas
import matplotlib
matplotlib.use('Agg')
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# tf.get_logger().setLevel('ERROR')

def train_PPO(config, exp_type=None, data_type=None, data_num=None):
    # tf.reset_default_graph()
    if config.agent_type == 'PPO1' or config.agent_type == 'PPO2':
        Agent = PPO
    if config.agent_type == 'PPORB':
        Agent = PPO_RB
    if config.agent_type == 'TRC':
        Agent = TRC
    
    # Setup for output directory and logging
    if exp_type is not None:
        if data_type is not None:
            if data_num is not None:
                config.data_path = 'datasets/{}/{}/{}'.format(exp_type, data_type, data_num)
                output_dir = 'output/{}/{}/{}/{}_{}_{}'.format(config.agent_type, exp_type, data_type,
                                                               config.clip_ratio, config.sigma, datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
            else:
                config.data_path = 'datasets/{}/{}'.format(exp_type, data_type)
                output_dir = 'output/{}/{}/{}/{}_{}'.format(config.agent_type, exp_type, data_type, config.score_type, datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        else:
            config.data_path = 'datasets/{}'.format(exp_type)
            output_dir = 'output/{}/{}/{}_{}'.format(config.agent_type, exp_type, config.encoder_type, datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        if int(exp_type[3])<=2:
            config.transpose = True
        else:
            config.transpose = False
    else:
        output_dir = 'output/{}/{}'.format(config.agent_type, datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir), level_str='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info('Python version is {}'.format(platform.python_version()))
    _logger.info('Current commit of code: ___')
    
    # Get running configuration
    config.save_model_path = '{}/model'.format(output_dir)
    config.summary_dir = '{}/summary'.format(output_dir)
    config.plot_dir = '{}/plot'.format(output_dir)
    config.graph_dir = '{}/graph'.format(output_dir)
    
    # Create directory
    create_dir(config.save_model_path)
    create_dir(config.summary_dir)
    create_dir(config.plot_dir)
    create_dir(config.graph_dir)
    set_seed(config.seed)
    
    # Log the configuration parameters
    _logger.info('Configuration parameters: {}'.format(vars(config)))    # Use vars to convert config to dict for logging
    
    try:
        file_path = '{}/data.npy'.format(config.data_path)
        solution_path = '{}/DAG.npy'.format(config.data_path)
        training_set = DataGenerator(file_path, solution_path, config.normalize, config.transpose)
    except ValueError:
        print ("Only support importing data from existing files")
        
    sl, su, strue = BIC_lambdas(training_set.inputdata, None, None, training_set.true_graph.T, config.reg_type, config.score_type)
    lambda1 = 0
    lambda1_upper = 5
    lambda1_update_add = 1
    lambda2 = 1/(10**(np.round(config.max_length/3)))
    lambda2_upper = 0.01
    lambda2_update_mul = 10
    lambda_iter_num = config.lambda_iter_num
    # test initialized score
    _logger.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
    _logger.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(sl, su, lambda2,
                 (strue-sl)/(su-sl)*lambda1_upper))
    
    agent = Agent(config)
    callreward = reward(agent.batch_size, config.max_length, agent.input_dimension, training_set.inputdata,
                        sl, su, lambda1_upper, config.score_type, config.reg_type, config.l1_graph_reg, False)

    _logger.info('Finished creating training dataset, agent model and reward class')
    
    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)  
    
    _logger.info('Starting session...')
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())
        _logger.info('Shape of agent.input: {}'.format(sess.run(tf.shape(agent.input_))))
        # _logger.info('training_set.true_graph: {}'.format(training_set.true_graph))
        # _logger.info('training_set.b: {}'.format(training_set.b))
    
        # Initialize useful variables
        advantages, rewards_batches, reward_max_per_batch, lambda1s, lambda2s = [],[],[],[],[]
        # graphss, probsss, max_rewards, accuracy_res, accuracy_res_pruned = [],[],[],[],[]
        max_rewards, accuracy_res, accuracy_res_pruned = [],[],[]
        max_reward = float('-inf')
        max_reward_score_cyc = (lambda1_upper+1, 0)
    
        # Summary writer
        writer = tf.summary.FileWriter(config.summary_dir, sess.graph)
        _logger.info('Starting training.')
        
        for i in (range(1, config.nb_epoch + 1)):
            if config.verbose:
                _logger.info('Start training for {}-th epoch'.format(i))

            input_batch = training_set.train_batch(agent.batch_size, agent.max_length, agent.input_dimension)
            graphs, log_pi, value_s, logits_for_rewards, = sess.run(
                [agent.graphs, agent.log_pi, agent.critic.predictions, agent.logits_for_rewards], feed_dict={agent.input_: input_batch})
            reward_cal = callreward.cal_rewards(graphs, lambda1, lambda2)
            advantage = -reward_cal[:,0] - value_s
            advantage_abs = advantage
            package = list(zip(input_batch, graphs, log_pi, -reward_cal[:,0], advantage, advantage_abs))
            agent.replayer.store(pd.DataFrame(package, columns=['input','graphs','log_pis','reward','advantage', 'advantage_abs']))
            
            # max reward, max reward per batch
            max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
            max_reward_batch = float('inf')
            max_reward_batch_score_cyc = (0, 0)
    
            for reward_, score_, cyc_ in reward_cal:
                if reward_ < max_reward_batch:
                    max_reward_batch = reward_
                    max_reward_batch_score_cyc = (score_, cyc_)
            max_reward_batch = -max_reward_batch
            if max_reward < max_reward_batch:
                max_reward = max_reward_batch
                max_reward_score_cyc = max_reward_batch_score_cyc
            
            # for average reward per batch
            reward_batch_score_cyc = np.mean(reward_cal[:,1:], axis=0)
            input_feed, graphs_feed, log_pis_feed, reward_feed, advantage_feed, _ = agent.replayer.sample()
            # Get feed dict
            feed = {agent.input_: input_batch, agent.log_pi_old_: log_pis_feed, 
                    agent.advantage_: advantage_feed, agent.reward_: reward_feed, agent.graphs_: graphs_feed}
            
            # PPO1
            if config.agent_type == 'PPO1':
                summary, probs, graph_batch, advantage, ratio, loss1, log_pi, log_pi_old, kl_loss, \
                surr, kl_loss, beta, train_step1, train_step2 = sess.run(
                        [agent.merged, agent.scores, agent.graph_batch, agent.advantage, agent.ratio, agent.loss1,
                         agent.log_pi, agent.log_pi_old, agent.surr, agent.kl_loss, agent.kl_loss, agent.beta, agent.train_step1, agent.train_step2], 
                        feed_dict=feed)
                if kl_loss.mean() > agent.d_tar/1.5:
                    agent.beta *= 2
                elif kl_loss.mean() < agent.d_tar/1.5:
                    agent.beta *= 0.5
            
            # PPO2
            if config.agent_type == 'PPO2' or config.agent_type == 'PPORB':
                summary, probs, graph_batch, advantage, \
                clipped_surrogate, train_step1, train_step2 = sess.run(
                        [agent.merged, agent.scores, agent.graph_batch, agent.advantage,
                         agent.clipped_surrogate, agent.train_step1, agent.train_step2], 
                        feed_dict=feed)

            # TRC
            if config.agent_type == 'TRC':
                summary, probs, graph_batch, advantage, \
                clipped_surrogate, kl_mat, log1, train_step1, train_step2 = sess.run(
                        [agent.merged, agent.scores, agent.graph_batch, agent.advantage,
                         agent.clipped_surrogate, agent.kl_mat, agent.log1, agent.train_step1, agent.train_step2], 
                        feed_dict=feed)
                
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)
            advantages.append(advantage)
            rewards_batches.append(reward_batch_score_cyc)
            reward_max_per_batch.append(max_reward_batch_score_cyc)
    
            # graphss.append(graph_batch)
            # probsss.append(probs)
            max_rewards.append(max_reward_score_cyc)
            
            print ('[iter {}] reward_batch: {}'.format(i, np.mean(reward_feed)))
            # logging
            if i == 1 or i % 100 == 0:
                if i >= 100:
                    writer.add_summary(summary,i)
                _logger.info('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'.format(i,
                             np.mean(reward_feed), max_reward, max_reward_batch))
    
                plt.figure(1)
                plt.plot(rewards_batches, label='reward per batch')
                plt.plot(max_rewards, label='max reward')
                plt.legend()
                plt.savefig('{}/reward_batch_average.png'.format(config.plot_dir))
                plt.close()
                    
            # update lambda1, lamda2
            if (i+1) % lambda_iter_num == 0:
                ls_kv = callreward.update_all_scores(lambda1, lambda2)
                # np.save('{}/solvd_dict_epoch_{}.npy'.format(config.graph_dir, i), np.array(ls_kv))
                max_rewards_re = callreward.update_scores(max_rewards, lambda1, lambda2)
                rewards_batches_re = callreward.update_scores(rewards_batches, lambda1, lambda2)
                reward_max_per_batch_re = callreward.update_scores(reward_max_per_batch, lambda1, lambda2)
    
                # saved somewhat more detailed logging info
                # np.save('{}/solvd_dict.npy'.format(config.graph_dir), np.array(ls_kv))
                pd.DataFrame(np.array(max_rewards_re)).to_csv('{}/max_rewards.csv'.format(output_dir))
                pd.DataFrame(rewards_batches_re).to_csv('{}/rewards_batch.csv'.format(output_dir))
                pd.DataFrame(reward_max_per_batch_re).to_csv('{}/reward_max_batch.csv'.format(output_dir))
                pd.DataFrame(lambda1s).to_csv('{}/lambda1s.csv'.format(output_dir))
                pd.DataFrame(lambda2s).to_csv('{}/lambda2s.csv'.format(output_dir))
                
                graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]
    
                if cyc_min < 1e-5:
                    lambda1_upper = score_min
                lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                _logger.info('[iter {}] lambda1 {}, upper {}, lambda2 {}, upper {}, score_min {}, cyc_min {}'.format(i+1,
                             lambda1, lambda1_upper, lambda2, lambda2_upper, score_min, cyc_min))
                    
                graph_batch = convert_graph_int_to_adj_mat(graph_int)
                if config.reg_type == 'LR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, training_set.inputdata))
                elif config.reg_type == 'QR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                elif config.reg_type == 'GPR':
                    graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))

                # estimate accuracy
                acc_est = count_accuracy(training_set.true_graph, graph_batch.T)
                acc_est2 = count_accuracy(training_set.true_graph, graph_batch_pruned.T)
                fdr, tpr, fpr, shd, nnz = acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], acc_est['shd'], \
                                          acc_est['pred_size']
                fdr2, tpr2, fpr2, shd2, nnz2 = acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], acc_est2['shd'], \
                                               acc_est2['pred_size']
                
                # this draw the average graph per batch. 
                # can be modified to draw the graph (with or w/o pruning) that has the best reward
                fig = plt.figure(2)
                fig.suptitle('Iteration: {}'.format(i))
                ax = fig.add_subplot(1, 2, 1)
                ax.set_title('recovered_graph')
                ax.imshow(np.around(graph_batch_pruned.T).astype(int),cmap=plt.cm.gray)
                ax = fig.add_subplot(1, 2, 2)
                ax.set_title('ground truth')
                ax.imshow(training_set.true_graph, cmap=plt.cm.gray)
                plt.savefig('{}/recovered_graph_iteration_{}.png'.format(config.plot_dir, i+1))
                plt.close()
                
                accuracy_res.append((fdr, tpr, fpr, shd, nnz))
                accuracy_res_pruned.append((fdr2, tpr2, fpr2, shd2, nnz2))
                
                np.save('{}/accuracy_res.npy'.format(output_dir), np.array(accuracy_res))
                np.save('{}/accuracy_res2.npy'.format(output_dir), np.array(accuracy_res_pruned))
                np.save('{}/graphs_{}.npy'.format(config.graph_dir, i+1), graph_batch_pruned.T)
                    
                _logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                _logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

            # Save the variables to disk
            # if i % max(1, int(config.nb_epoch / 2)) == 0 and i != 0:
            #     curr_model_path = saver.save(sess, '{}/tmp.ckpt'.format(config.save_model_path), global_step=i)
            #     _logger.info('Model saved in file: {}'.format(curr_model_path))
                
        _logger.info('Training COMPLETED !')
        # saver.save(sess, '{}/agent.ckpt'.format(config.save_model_path))
    return