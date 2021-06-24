import os
import logging
import platform
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from models import A2C
from environment import Reward_BIC as reward
from utils.data_loader import DataGenerator
from utils.dir_utils import create_dir
from utils.log_utils import LogHelper
from utils.tf_utils import set_seed
from utils.analyze_utils import convert_graph_int_to_adj_mat, graph_prunned_by_coef, count_accuracy, graph_prunned_by_coef_2nd
from utils.cam_with_pruning_cam import pruning_cam
from utils.lambda_utils import BIC_lambdas
import matplotlib
matplotlib.use('Agg')

def train_A2C(config, exp_type=None, data_type=None, data_num=0):
    Agent = A2C
    
    # Setup for output directory and logging
    if exp_type is not None:
        if data_type is not None:
            config.data_path = 'datasets/{}/{}/{}'.format(exp_type, data_type, data_num)
            output_dir = 'output/{}/{}/{}/{}'.format(config.agent_type, exp_type, data_type, datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        else:
            config.data_path = 'datasets/{}'.format(data_type)
            output_dir = 'output/{}/{}/{}'.format(config.agent_type, data_type, datetime.now(timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        if int(exp_type[3])==1:
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
        rewards_avg_baseline = []
        rewards_batches = []
        reward_max_per_batch = []
        lambda1s = []
        lambda2s = []
        graphss = []
        probsss = []
        max_rewards = []
        max_reward = float('-inf')
        image_count = 0
        accuracy_res = []
        accuracy_res_pruned = []
        max_reward_score_cyc = (lambda1_upper+1, 0)
    
        # Summary writer
        writer = tf.summary.FileWriter(config.summary_dir, sess.graph)
        _logger.info('Starting training.')
        
        for i in (range(1, config.nb_epoch + 1)):
            if config.verbose:
                _logger.info('Start training for {}-th epoch'.format(i))
    
            input_batch = training_set.train_batch(agent.batch_size, agent.max_length, agent.input_dimension)
            graphs_feed, comb = sess.run([agent.graphs, agent.comb], feed_dict={agent.input_: input_batch})
            reward_feed = callreward.cal_rewards(graphs_feed, lambda1, lambda2)
            advantage = -reward_feed[:,0] - comb
            # max reward, max reward per batch
            max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
            max_reward_batch = float('inf')
            max_reward_batch_score_cyc = (0, 0)
    
            for reward_, score_, cyc_ in reward_feed:
                if reward_ < max_reward_batch:
                    max_reward_batch = reward_
                    max_reward_batch_score_cyc = (score_, cyc_)
                        
            max_reward_batch = -max_reward_batch
    
            if max_reward < max_reward_batch:
                max_reward = max_reward_batch
                max_reward_score_cyc = max_reward_batch_score_cyc
    
            # for average reward per batch
            reward_batch_score_cyc = np.mean(reward_feed[:,1:], axis=0)
                              
            if config.verbose:
                _logger.info('Finish calculating reward for current batch of graph')
            advantage = np.abs(advantage)
            package = list(zip(input_batch, graphs_feed, -reward_feed[:,0], advantage))
            agent.replayer.store(pd.DataFrame(package, columns=['input','log_pis','reward','advantage_abs']))
            input_batch, graphs_feed, in_reward_feed, advantage = agent.replayer.sample()
            # Get feed dict
            feed = {agent.input_: input_batch, agent.reward_: in_reward_feed, agent.graphs_:graphs_feed}
    
            summary, base_op, score_test, probs, graph_batch, \
                reward_batch, reward_avg_baseline, train_step1, train_step2 = sess.run(
                    [agent.merged, agent.base_op, agent.test_scores, agent.log_softmax, 
                     agent.graph_batch, agent.reward_batch, agent.avg_baseline, 
                     agent.train_step1, agent.train_step2], feed_dict=feed)
    
            if config.verbose:
                _logger.info('Finish updating agent and critic network using reward calculated')
                    
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)
    
            rewards_avg_baseline.append(reward_avg_baseline)
            rewards_batches.append(reward_batch_score_cyc)
            reward_max_per_batch.append(max_reward_batch_score_cyc)
    
            graphss.append(graph_batch)
            probsss.append(probs)
            max_rewards.append(max_reward_score_cyc)
            print ('[iter {}] reward_batch: {}'.format(i, reward_batch))
            # print ('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'.format(i, reward_batch, max_reward, max_reward_batch))
            # logging
            if i == 1 or i % 100 == 0:
                if i >= 100:
                    writer.add_summary(summary,i)
                _logger.info('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'.format(i,
                             reward_batch, max_reward, max_reward_batch))
                # other logger info; uncomment if you want to check
                # _logger.info('graph_batch_avg: {}'.format(graph_batch))
                # _logger.info('graph true: {}'.format(training_set.true_graph))
                # _logger.info('graph weights true: {}'.format(training_set.b))
                # _logger.info('=====================================')
    
                plt.figure(1)
                plt.plot(rewards_batches, label='reward per batch')
                plt.plot(max_rewards, label='max reward')
                plt.legend()
                plt.savefig('{}/reward_batch_average.png'.format(config.plot_dir))
                plt.close()
    
                image_count += 1
                # this draw the average graph per batch. 
                # can be modified to draw the graph (with or w/o pruning) that has the best reward
                fig = plt.figure(2)
                fig.suptitle('Iteration: {}'.format(i))
                ax = fig.add_subplot(1, 2, 1)
                ax.set_title('recovered_graph')
                ax.imshow(np.around(graph_batch.T).astype(int),cmap=plt.cm.gray)
                ax = fig.add_subplot(1, 2, 2)
                ax.set_title('ground truth')
                ax.imshow(training_set.true_graph, cmap=plt.cm.gray)
                plt.savefig('{}/recovered_graph_iteration_{}.png'.format(config.plot_dir, image_count))
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
                    # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                    # so we need to do a tranpose on the input graph and another tranpose on the output graph
                    graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))
    
                # estimate accuracy
                acc_est = count_accuracy(training_set.true_graph, graph_batch.T)
                acc_est2 = count_accuracy(training_set.true_graph, graph_batch_pruned.T)
    
                fdr, tpr, fpr, shd, nnz = acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], acc_est['shd'], \
                                          acc_est['pred_size']
                fdr2, tpr2, fpr2, shd2, nnz2 = acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], acc_est2['shd'], \
                                               acc_est2['pred_size']
                    
                accuracy_res.append((fdr, tpr, fpr, shd, nnz))
                accuracy_res_pruned.append((fdr2, tpr2, fpr2, shd2, nnz2))
                
                np.save('{}/accuracy_res.npy'.format(output_dir), np.array(accuracy_res))
                np.save('{}/accuracy_res2.npy'.format(output_dir), np.array(accuracy_res_pruned))
                    
                _logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                _logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))
    
            # Save the variables to disk
            if i % max(1, int(config.nb_epoch / 2)) == 0 and i != 0:
                curr_model_path = saver.save(sess, '{}/tmp.ckpt'.format(config.save_model_path), global_step=i)
                _logger.info('Model saved in file: {}'.format(curr_model_path))
    
        _logger.info('Training COMPLETED !')
        saver.save(sess, '{}/agent.ckpt'.format(config.save_model_path))
    return
