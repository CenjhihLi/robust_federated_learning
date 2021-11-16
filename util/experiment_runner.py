import itertools
import json
#import os
import pathlib
import random
import collections
import gc
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import tensorflow as tf

from util.aggregators import mean, median, trimmed_mean, gamma_mean, geometric_median
import prepare_data.emnist as emnist
import prepare_data.mnist as mnist
import prepare_data.fashion_mnist as fashion_mnist
import prepare_data.pneumonia as pneumonia
from util.client import Client
from util.server import Server
#nest_asyncio.apply()


# reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
def fs_setup(experiment_name, seed, config):
    """
    Setup the experiments fold and use config.json to record 
    the parameters of experiment
    This will use in run_experiment
    very useful since the experiment always stop...
    """
    root_dir = pathlib.Path(f'experiments') / experiment_name
    #root_dir = os.path.join(f'experiments',experiment_name)
    config_path = root_dir / 'config.json'
    #config_path = os.path.join(root_dir, 'config.json')

    # get model config
    if config_path.is_file():
        with config_path.open() as f:
            stored_config = json.load(f)
          
            if json.dumps(stored_config, sort_keys=True) != json.dumps(config, sort_keys=True):
                with (root_dir / 'config_other.json').open(mode='w') as f_other:
                    json.dump(config, f_other, sort_keys=True, indent=2)
                raise Exception('stored config should equal run_experiment\'s parameters')
    else:
        root_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open(mode='w') as f:
            json.dump(config, f, sort_keys=True, indent=2)
          
    experiment_dir = root_dir / f'seed_{seed}'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return experiment_dir


"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
"""
def run_experiment(experiment_name, seed, model_factory, input_shape, server_config,
                   partition_config, dataset, num_of_rounds, threat_model, initialize):
    server = Server(model_factory, **server_config, initialize=initialize)

    experiment_dir = fs_setup(experiment_name, seed, {
        # 'model': server.model.get_config(),
        'partition_config': partition_config
    })
    expr_basename = f'{server_config["weight_delta_aggregator"].__name__}' \
                  f'_cpr_{server_config["clients_per_round"]}' \
                  f'{(threat_model.prefix if threat_model is not None else "")}'
    expr_file = experiment_dir / f'{expr_basename}.npz'
    
    """
    The following part allow the experiment continue from the last stop round
    in server.train: for round in range(start_round,num_of_rounds)
        store server_weights&history in each iteration
    """
    if expr_file.is_file():
        prev_results = np.load(expr_file, allow_pickle=True)
        server_weights = prev_results['server_weights'].tolist()
        server.model.set_weights(server_weights)
        history = prev_results['history'].tolist()
        history_delta_sum = prev_results['history_delta_sum'].tolist()
        #last_deltas = prev_results['last_deltas'].tolist()
        start_round = len(history)
        if start_round >= num_of_rounds:
            print(f'skipping {expr_basename} (seed={seed}) '
                  f'start_round({start_round}), num_of_rounds({num_of_rounds})')
            return
    else:
        history_delta_sum = []
        history = []
        #last_deltas = []
        start_round = 0

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    x_chest = False
    val_x, val_y = None, None
    if dataset == "emnist":
        """
        Use emnist dataset provided by tff:
        https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data?hl=zh-tw
        """
        train_data, test_x, test_y  = emnist.load(client = partition_config['#clients'],
                                                  reshape = input_shape)
        optimizer = tf.keras.optimizers.SGD
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        initial_lr = 5e-2
    elif dataset == "mnist":
        train_data, (test_x, test_y) = mnist.load(partition_config, input_shape) 
        optimizer = tf.keras.optimizers.SGD
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        initial_lr = 1e-1
    elif dataset == "fashion_mnist":
        train_data, (test_x, test_y) = fashion_mnist.load(partition_config, input_shape)
        optimizer = tf.keras.optimizers.SGD
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        initial_lr = 5e-2
    elif dataset == "pneumonia":
        """
        Use pneumonia dataset provided by:
        https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
        """
        train_data, (val_x, val_y), (test_x, test_y) = pneumonia.load(partition_config, input_shape)
        optimizer = tf.keras.optimizers.Adam
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        initial_lr = 1e-4
        x_chest = True
    
    clients = [
        Client(i, data, model_factory)
        for i, data in enumerate(train_data)
        ]
    if threat_model is not None:
        attackers = np.random.choice(clients, int(
            len(clients) * threat_model.real_alpha) if threat_model.real_alpha is not None else threat_model.f, replace=False)
        for client in attackers:
            client.as_attacker(threat_model)

    del train_data, threat_model
    gc.collect()

    server.train(clients, val_x, val_y, test_x, test_y, start_round, num_of_rounds, expr_basename, history, history_delta_sum,
                 x_chest, optimizer, loss_fn, initial_lr, 
                 lambda history, server_weights, history_delta_sum: np.savez(expr_file, history=history, 
                    server_weights=server_weights, history_delta_sum=history_delta_sum))
    del server, clients, test_x, test_y, start_round, num_of_rounds, expr_basename, history, history_delta_sum, optimizer, loss_fn, initial_lr
    tf.keras.backend.clear_session()
    gc.collect()
    #server.train(seed, clients, test_x, test_y, start_round, num_of_rounds, expr_basename, history, history_delta_sum, last_deltas,
    #             lambda history, server_weights, history_delta_sum, last_deltas: np.savez(expr_file, history=history, 
    #                server_weights=server_weights, history_delta_sum=history_delta_sum, last_deltas = last_deltas))
    #for adam


@dataclass(frozen=True)
class Threat_model:
  type: str
  num_samples_per_attacker: int
  real_alpha: int = None
  f: int = None
  prefix: str = field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'prefix',
                       f'_b_{self.type}_'
                       f'{int(self.real_alpha * 100) if self.real_alpha is not None else "f" + str(self.f)}_'
                       f'{self.num_samples_per_attacker}')

def run_all(experiment, model_factory, input_shape, 
            seed, cpr, rounds, mu, sigma, dataset,
            real_alpha, num_samples_per_attacker=1_000_000, attack_type='random',
            t_mean_beta=0.1, real_alpha_as_f=False,
            gam_max=10, gamma=0.1, geo_max=1000, tol = 1e-7, clients = 20, initialize = None):
    if (real_alpha>1) or (real_alpha<0):
        raise ValueError("The proportion of attacker (real_alpha) should be in [0,1]")
    t_mean = partial(trimmed_mean, beta=t_mean_beta)
    t_mean.__name__ = f't_mean_{int(t_mean_beta * 100)}'
    
    r_gam_mean_s = partial(gamma_mean, gamma = gamma, max_iter=gam_max, tol = tol, compute='simple')
    r_gam_mean_s.__name__ = 'simple_record_gamma_mean_{}'.format(str(gamma).replace(".","_"))
  
    r_gam_mean = partial(gamma_mean, gamma = gamma, max_iter=gam_max, tol = tol)
    r_gam_mean.__name__ = 'record_gamma_mean_{}'.format(str(gamma).replace(".","_"))

    gam_mean_s = partial(gamma_mean, gamma = gamma, max_iter=gam_max, tol = tol, compute='simple')
    gam_mean_s.__name__ = 'simple_gamma_mean_{}'.format(str(gamma).replace(".","_"))
  
    gam_mean = partial(gamma_mean, gamma = gamma, max_iter=gam_max, tol = tol)
    gam_mean.__name__ = 'gamma_mean_{}'.format(str(gamma).replace(".","_"))
  
    geo_mean = partial(geometric_median, max_iter = geo_max, tol = tol)
    geo_mean.__name__ = 'geometric_median'
  
    weight_delta_aggregators = [mean, median, r_gam_mean_s, r_gam_mean, gam_mean_s, gam_mean, geo_mean, t_mean]
    #weight_delta_aggregators = [r_gam_mean_s, gam_mean_s, geo_mean, t_mean, median, median, mean]

    threat_models = [None] if (attack_type is None or real_alpha==0) else [
        Threat_model(type=attack_type, num_samples_per_attacker=num_samples_per_attacker,
                     f=real_alpha) if real_alpha_as_f else Threat_model(type=attack_type,
                                num_samples_per_attacker=num_samples_per_attacker,
                                real_alpha=real_alpha),
                        ]

    for (threat_model, wda) in itertools.product(threat_models, weight_delta_aggregators):
        run_experiment(experiment,
                        seed=seed,
                        model_factory=model_factory,
                        input_shape = input_shape,
                        server_config={
                           'weight_delta_aggregator': wda,
                           'clients_per_round': cpr,
                           },
                        dataset = dataset,
                        partition_config={'#clients': clients, 'mu': mu, 'sigma': sigma},
                        num_of_rounds=rounds,
                        threat_model=threat_model,
                        initialize = initialize
                        )
