from utils.config_utils import get_config
import matplotlib
import multiprocessing as mp
import warnings
import time
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
matplotlib.use('Agg')

config, _ = get_config()

        
# experiment 0: exp1_12nodes
exp1_12nodes = 'exp1_12nodes'
exp1_12nodes_list = ['gauss_diff_noise','gauss_same_noise','lingam_diff_noise','lingam_same_noise']
# exp1_12nodes_list = ['lingam_diff_noise','lingam_same_noise']
exp1_12nodes_list_BIC = ['lingam_diff_noise']
exp1_12nodes_range = ['1','2','3','4','5']
# experiment 1: exp1_30nodes
exp1_30nodes = 'exp1_30nodes_edge_prob_02'
exp1_30nodes_list = ['1','2','3','4','5']
# experiment 2: exp2_quad
exp2_quad = 'exp2_10_nodes_quad'
prefix = 'lingam_quad_same_noise_seed'
exp2_quad_list = [prefix+'8',prefix+'100',prefix+'500',prefix+'1231',prefix+'1280'] 
# experiment 3: exp3_gp
exp3_gp = 'exp3_10_nodes_gp'
exp3_gp_list = ['1','2','3','4','5']
# experiment 4: exp4
exp4 = 'exp4_sachs'
exp6 = 'exp6_child'
experiment_dict = {0:exp1_12nodes, 1:exp1_30nodes, 2:exp2_quad, 3:exp3_gp, 4:exp4, 6:exp6}
experiment_sub_list = {1:exp1_30nodes_list, 2:exp2_quad_list, 3:exp3_gp_list}
experiment = 0
sub_exp_1 = 0
exp4_times = 30

def add_config(config, experiment):
    if experiment == 0:
        config.max_length = 12
        config.data_size = 5000
        config.reg_type = 'LR' 
        config.transpose = True
        config.input_dimension = 64
    if experiment == 1:
        config.max_length = 30
        config.data_size = 5000
        config.score_type = 'BIC'
        config.reg_type = 'LR' 
        config.use_bias  = True
        config.bias_initial_value = -10.0
        config.batch_size = 128
        config.transpose = True
        config.input_dimension = 128
        config.nb_epoch = 40000
    if experiment == 2:
        config.max_length = 10
        config.data_size = 3000
        # config.score_type = 'BIC'
        config.score_type = 'BIC_different_var'
        config.reg_type = 'QR' 
        config.transpose = True
        config.input_dimension = 64
        config.nb_epoch = 30000
    if experiment == 3:
        config.max_length = 10
        config.data_size = 1000
        config.score_type = 'BIC'
        config.reg_type = 'GPR'
        config.normalize = True
        config.transpose = False
        config.input_dimension = 128
        config.nb_epoch = 20000
    if experiment == 4:
        config.max_length = 11
        config.data_size = 853
        config.score_type = 'BIC_different_var'
        # config.score_type = 'BIC'
        config.reg_type = 'GPR'
        config.use_bias  = False
        config.bias_initial_value = -10.
        config.normalize = True
        config.transpose = False
        config.input_dimension = 128
        config.nb_epoch = 20000
    if experiment == 6:
        config.encoder_type = 'TransformerEncoder'
        config.num_heads = 16
        config.agent_type = 'VPG'
        config.max_length = 20
        config.data_size = 500
        config.score_type = 'BIC_different_var'
        # config.score_type = 'BIC'
        config.reg_type = 'GPR'
        config.use_bias  = False
        config.bias_initial_value = -10.
        config.normalize = True
        config.transpose = False
        config.input_dimension = 128
        config.nb_epoch = 20000
    return config

# prob_enc = 2
config = add_config(config, experiment)
# sleep_time = 60
# if random.randint(1,4) % 4 == 0:
#     config.score_type = 'BIC'
# if random.randint(1,4) % 4 == 0:
#     config.use_bias = True

# randa = random.randint(1, prob_enc)
# if randa == 2: config.agent_type = 'TRC'
# randb = random.randint(1, prob_enc)
# if randb == 2: config.score_type = 'BIC'

config.agent_type = 'PPO2'
if 'VPG' in config.agent_type:
    from train_VPG import train_VPG
    train_type = train_VPG
elif config.agent_type == 'A2C':
    from train_A2C import train_A2C
    train_type = train_A2C
else:
    from train_PPO import train_PPO
    train_type = train_PPO
# if randa == 1:
#     config.num_gat_stacks = 6
# elif randa == 2:
#     config.num_gat_stacks = 8
# else:
#     config.num_gat_stacks = 10
# config.num_gat_stacks = random.randint(2, 5)*2
train_type(config, exp1_12nodes, exp1_12nodes_list[sub_exp_1], exp1_12nodes_range[0])
# train_type(config, experiment_dict[experiment], experiment_sub_list[experiment][0])
# time.sleep(sleep_time)

# if __name__ == '__main__':
#     time0 = time.time()
#     record = []
#     if experiment == 0:
#         if sub_exp_1 == None:
#             config = add_config(config, experiment)
#             for i in exp1_12nodes_list:
#                 record = []
#                 print ('Training on {}'.format(i))
#                 for j in range(5):
#                     process = mp.Process(target=train_type, args=(config, exp1_12nodes, i, exp1_12nodes_range[j]))
#                     process.start()
#                     record.append(process)
#                 for process in record:
#                     process.join()
#         else:
#             record = []
#             config = add_config(config, experiment)
#             for j in range(5):
#                 process = mp.Process(target=train_type, args=(config, exp1_12nodes, exp1_12nodes_list[sub_exp_1], exp1_12nodes_range[j]))
#                 process.start()
#                 record.append(process)
#             for process in record:
#                 process.join()

#     if experiment in [1,2]:
#         from train_VPG import train_VPG
#         train_type = train_VPG
#         config = add_config(config, experiment)
#         config.agent_type = 'VPG_Replay'
#         # config.agent_type = 'VPG'
#         for j in range(5):
#             process = mp.Process(target=train_type, args=(config, experiment_dict[experiment], experiment_sub_list[experiment][j]))
#             process.start()
#             record.append(process)
#         for process in record:
#             process.join()
    
#     if experiment == 3:
#         config = add_config(config, experiment)
#         train_type(config, experiment_dict[experiment], experiment_sub_list[experiment][0])
    
#     if experiment == 4:
#         config = add_config(config, experiment)
#         for j in range(exp4_times):
#             process = mp.Process(target=train_type, args=(config, experiment_dict[experiment]))
#             process.start()
#             record.append(process)
#         for process in record:
#             process.join()

#     time1 = time.time()
#     seconds = int(time1 - time0)
#     hours = seconds // 3600
#     mins = (seconds-hours*3600) // 60
#     seconds -= hours * 3600 + mins * 60
#     print ('Total Training Time: {} hour, {} min, {} sec'.format(hours, mins, seconds))
    
# if __name__ == '__main__':
#     time0 = time.time()
#     record = []
#     for i in exp1_12nodes_list_BIC:
#         record = []
#         print ('Training on {}'.format(i))
#         config = add_config(config, 0)
#         config.score_type = 'BIC'
#         for j in range(5):
#             process = mp.Process(target=train_type, args=(config, exp1_12nodes, i, exp1_12nodes_range[j]))
#             process.start()
#             record.append(process)
#         for process in record:
#             process.join()
#         time.sleep(sleep_time)
    
#     for i in exp1_12nodes_list:
#         record = []
#         print ('Training on {}'.format(i))
#         config = add_config(config, 0)
#         config.score_type = 'BIC_different_var'
#         for j in range(5):
#             process = mp.Process(target=train_type, args=(config, exp1_12nodes, i, exp1_12nodes_range[j]))
#             process.start()
#             record.append(process)
#         for process in record:
#             process.join()
#         time.sleep(sleep_time)
    
#     for experiment in [1,2]:
#         record = []
#         config = add_config(config, experiment)
#         for j in range(5):
#             process = mp.Process(target=train_type, args=(config, experiment_dict[experiment], experiment_sub_list[experiment][j]))
#             process.start()
#             record.append(process)
#         for process in record:
#             process.join()
#         time.sleep(sleep_time)
    
#     for j in range(exp4_times):
#         record = []
#         config = add_config(config, experiment)
#         process = mp.Process(target=train_type, args=(config, experiment_dict[experiment], experiment_sub_list[experiment][j]))
#         process.start()
#         record.append(process)
#     for process in record:
#         process.join()
        
#     experiment = 3
#     train_type(config, experiment_dict[experiment], experiment_sub_list[experiment][0])
        
#     time1 = time.time()
#     seconds = int(time1 - time0)
#     hours = seconds // 3600
#     mins = (seconds-hours*3600) // 60
#     seconds -= hours * 3600 + mins * 60
#     print ('Total Training Time: {} hour, {} min, {} sec'.format(hours, mins, seconds))