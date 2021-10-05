#import sys
#sys.path.append("C:/GitHub\robust_federated_learning")
import tensorflow.keras as keras
import util.experiment_runner as experiment_runner
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.applications import resnet50, xception

seed=1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

we using some useful function refer from their code (but change quite much) and apply our gamma mean and geometric median as aggregators
"""

clients=20
def mlp_model_factory():
    return keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
        ])
#model_factory = mlp_model_factory

def cnn_model_factory():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, 
                            input_shape=(28, 28, 1), activation='relu', 
                            padding='same'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
        ])

#model_factory = cnn_model_factory
#input_shape = [28,28,1]
#model_factory = mlp_model_factory
#input_shape = [-1]

def res_model_factory():
    model=resnet50.ResNet50(input_shape=(150,150,3),weights='imagenet',include_top=False, )
    inputs = keras.Input(shape=(150, 150, 3))
    x = model(inputs, training=False)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256,activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1,activation='sigmoid')(x)
    return keras.Model(inputs, x)

experiment_runner.run_all('expr_no_attacks',
                          mlp_model_factory, input_shape=[-1], dataset='mnist',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0,
                          t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('expr_random',
                          mlp_model_factory, input_shape=[-1], dataset='mnist', 
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('cnn_expr_no_attacks',
                          model_factory = cnn_model_factory, input_shape=[28,28,1], dataset='mnist',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, 
                          real_alpha=0, t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('cnn_expr_random', 
                          model_factory = cnn_model_factory, input_shape=[28,28,1], dataset='mnist',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('fashion_mnist_cnn_expr_no_attacks',
                          model_factory = cnn_model_factory, input_shape=[28,28,1], dataset='fashion_mnist',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, 
                          real_alpha=0, t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('fashion_mnist_cnn_expr_random', 
                          model_factory = cnn_model_factory, input_shape=[28,28,1], dataset='fashion_mnist',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('fashion_mnist_expr_no_attacks',
                          mlp_model_factory, input_shape=[-1], dataset='fashion_mnist',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0,
                          t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('fashion_mnist_expr_random',
                          mlp_model_factory, input_shape=[-1], dataset='fashion_mnist', 
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('pneumonia_expr_no_attacks',
                          res_model_factory, input_shape=[150,150,3], dataset='pneumonia',
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0,
                          t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('pneumonia_expr_random',
                          res_model_factory, input_shape=[150,150,3], dataset='pneumonia', 
                          seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1, clients=clients,
                          gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

