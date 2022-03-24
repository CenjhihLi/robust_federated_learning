#import sys
#sys.path.append("C:/GitHub\robust_federated_learning")
import util.experiment_runner as experiment_runner
import numpy as np
import tensorflow as tf
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
seed=1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
from util.model import mlp_model_factory, cnn_model_factory, LeNet_model_factory, LeNet5_model_factory, res_model, CNN_model_xray, CNN_model_xray_initialize
"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

we use some useful function refer from their code (but change quite a lot) and apply our gamma mean and geometric median as aggregators
"""

if __name__ == '__main__':
    clients=20
    partition_config={'#clients': clients, 'mu': 1.5, 'sigma': 3.45, 'min_value': 0}
    # model = mlp_model_factory
    # input_shape = [-1]
    # experiment_runner.run_all('expr_MNIST_mlp_no_attacks',
    #                           model, input_shape=input_shape, dataset='mnist',
    #                           seed=seed, cpr='all', rounds=1000, real_alpha=0, partition_config=partition_config,
    #                           t_mean_beta=0.1, clients=clients,
    #                           gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    # experiment_runner.run_all('expr_MNIST_mlp_random',
    #                           model, input_shape=input_shape, dataset='mnist', 
    #                           seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
    #                           num_samples_per_attacker=1_000_000, 
    #                           attack_type='random', t_mean_beta=0.1, clients=clients,
    #                           gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    # model = cnn_model_factory
    # input_shape=[28,28,1]
    # experiment_runner.run_all('expr_MNIST_cnn_no_attacks',
    #                           model_factory = model, input_shape=input_shape, dataset='mnist',
    #                           seed=seed, cpr='all', rounds=1000, real_alpha=0,  partition_config=partition_config,
    #                           t_mean_beta=0.1, clients=clients,
    #                           gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    # experiment_runner.run_all('expr_MNIST_cnn_random', 
    #                           model_factory = model, input_shape=input_shape, dataset='mnist',
    #                           seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
    #                           num_samples_per_attacker=1_000_000, 
    #                           attack_type='random', t_mean_beta=0.1, clients=clients,
    #                           gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    # experiment_runner.run_all('expr_fashion_cnn_no_attacks',
    #                           model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
    #                           seed=seed, cpr='all', rounds=1000, real_alpha=0,  partition_config=partition_config,
    #                           t_mean_beta=0.1, clients=clients,
    #                           gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    # experiment_runner.run_all('expr_fashion_cnn_random', 
    #                           model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
    #                           seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
    #                           num_samples_per_attacker=1_000_000, 
    #                           attack_type='random', t_mean_beta=0.1, clients=clients,
    #                           gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)
    
    model = LeNet_model_factory #simplified, only 1 FC layer
    input_shape=[28,28,1]
    method_list = ['mean', 'median', 'geo_mean', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_MNIST_LeNet_random', 
                              model_factory = model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
                              num_samples_per_attacker=1_000_000, method_list = method_list, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)
    
    experiment_runner.run_all('final_expr_MNIST_LeNet_no_attacks',
                              model_factory = model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0,  partition_config=partition_config,
                              method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)
    
    experiment_runner.run_all('final_expr_fashion_LeNet_random', 
                              model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
                              num_samples_per_attacker=1_000_000, method_list = method_list, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)
    
    experiment_runner.run_all('final_expr_fashion_LeNet_no_attacks',
                              model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0,  partition_config=partition_config,
                              method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)
    
    res_model_factory = res_model(pooling = 'avg') 
    clients = 9
    partition_config={'#clients': clients, 'mu': 1.5, 'sigma': 3.45, 'min_value': 512} #seed 1: []
    
    model = res_model_factory
    input_shape = [150,150,3]
    initialize = None

    method_list = ['median', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_pneumonia_selfval_resnet_avg_9client_random',
                              model, input_shape=input_shape, dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, real_alpha=0.2, partition_config=partition_config,
                              self_split_val = True, epochs=20, method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7, initialize = initialize)

    method_list = ['mean', 'median', 'geo_mean', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_pneumonia_selfval_resnet_avg_9client_no_attack',
                              model, input_shape=input_shape, dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, real_alpha=0, partition_config=partition_config,
                              self_split_val = True, epochs=20, method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7, initialize = initialize)     
    
    clients = 5
    partition_config={'#clients': clients, 'mu': 1.5, 'sigma': 3.45, 'min_value': 900} #seed 1: []
    
    model = res_model_factory
    input_shape = [150,150,3]
    initialize = None

    method_list = ['median', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_pneumonia_selfval_resnet_avg_5client_random',
                              model, input_shape=input_shape, dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, real_alpha=0.2, partition_config=partition_config,
                              self_split_val = True, epochs=20, method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7, initialize = initialize)

    method_list = ['mean', 'median', 'geo_mean', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_pneumonia_selfval_resnet_avg_5client_no_attack',
                              model, input_shape=input_shape, dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, real_alpha=0, partition_config=partition_config,
                              self_split_val = True, epochs=20, method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7, initialize = initialize) 
    
    clients=20
    partition_config={'#clients': clients, 'mu': 1.5, 'sigma': 3.45, 'min_value': 0}
    model = LeNet_model_factory #simplified, only 1 FC layer
    input_shape=[28,28,1]
    method_list = ['mean', 'median', 'geo_mean', 'gam_mean_s_median', 'gam_mean_median']    
    # gamma = 2/931080
    experiment_runner.run_all('final_expr_MNIST_LeNet_random', 
                              model_factory = model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
                              num_samples_per_attacker=1_000_000, method_list = method_list, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=2/931080, geo_max=1000, tol = 1e-7)
    
    experiment_runner.run_all('final_expr_MNIST_LeNet_no_attacks',
                              model_factory = model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0,  partition_config=partition_config,
                              method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=2/931080, geo_max=1000, tol = 1e-7)
    
    experiment_runner.run_all('final_expr_fashion_LeNet_random', 
                              model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0.1, partition_config=partition_config,
                              num_samples_per_attacker=1_000_000, method_list = method_list, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=2/931080, geo_max=1000, tol = 1e-7)
    
    experiment_runner.run_all('final_expr_fashion_LeNet_no_attacks',
                              model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, real_alpha=0,  partition_config=partition_config,
                              method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=2/931080, geo_max=1000, tol = 1e-7)
    
    res_model_factory = res_model(pooling = 'avg') 
    clients = 9
    partition_config={'#clients': clients, 'mu': 1.5, 'sigma': 3.45, 'min_value': 512} #seed 1: []
    
    model = res_model_factory
    input_shape = [150,150,3]
    initialize = None
    
    # gamma = 2/543809
    method_list = ['median', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_pneumonia_selfval_resnet_avg_9client_random',
                              model, input_shape=input_shape, dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, real_alpha=0.2, partition_config=partition_config,
                              self_split_val = True, epochs=20, method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=2/543809, geo_max=1000, tol = 1e-7, initialize = initialize)

    method_list = ['mean', 'median', 'geo_mean', 'gam_mean_s_median', 'gam_mean_median']
    experiment_runner.run_all('final_expr_pneumonia_selfval_resnet_avg_9client_no_attack',
                              model, input_shape=input_shape, dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, real_alpha=0, partition_config=partition_config,
                              self_split_val = True, epochs=20, method_list = method_list, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=2/543809, geo_max=1000, tol = 1e-7, initialize = initialize) 