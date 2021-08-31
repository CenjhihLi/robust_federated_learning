import sys
sys.path.append("C:/GitHub\Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting-main")
import tensorflow.keras as keras
import experiment_runner as experiment_runner

"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

we using the experiments code and apply our gamma mean as an aggregator

Author: Cen-Jhih Li
Belongs: Academic Senica, Institute of statistic, Robust federated learning project
"""

clients=100
def mlp_model_factory():
    return keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
        ])
#model_factory = mlp_model_factory

# Wrap a Keras model for use with TFF.
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

experiment_runner.run_all('expr_no_attacks',
                          mlp_model_factory, input_shape=[-1], dataset='mnist',
                          seed=1, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0,
                          t_mean_beta=0.1,
                          gam_max=10, gamma=0.05, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('expr_random',
                          mlp_model_factory, input_shape=[-1], dataset='mnist', 
                          seed=1, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1,
                          gam_max=10, gamma=0.05, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('cnn_expr_no_attacks',
                          model_factory = cnn_model_factory, input_shape=[28,28,1], dataset='mnist',
                          seed=1, cpr='all', rounds=1000, mu=1.5, sigma=3.45, 
                          real_alpha=0, t_mean_beta=0.1, 
                          gam_max=10, gamma=0.05, geo_max=1000, tol = 1e-7)

experiment_runner.run_all('cnn_expr_random', 
                          model_factory = cnn_model_factory, input_shape=[28,28,1], dataset='mnist',
                          seed=1, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                          num_samples_per_attacker=1_000_000, 
                          attack_type='random', t_mean_beta=0.1,
                          gam_max=10, gamma=0.05, geo_max=1000, tol = 1e-7)
