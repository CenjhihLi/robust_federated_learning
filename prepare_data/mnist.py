#reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
import random

import numpy as np
import tensorflow as tf

from prepare_data.partition_v2 import Partition, PartitionParams


def load(partition_config, input_shape):
  (x_train, y_train), (x_test, y_test) = [
    (
      #np.divide(x.reshape((x.shape[0], -1)), 255., dtype=np.float32),
      np.divide(x.reshape([x.shape[0]]+input_shape), 255., dtype=np.float32),
      y.astype(np.int32).reshape((y.shape[0], -1))
    )
    for x, y in tf.keras.datasets.mnist.load_data()
  ]

  partition = Partition.random_log_normal_partition(
    PartitionParams(
      mu=partition_config['mu'],
      sigma=partition_config['sigma'],
      k=partition_config['#clients'],
      n=x_train.shape[0],
    ))

  shuffled_ds = list(zip(x_train, y_train))
  random.shuffle(shuffled_ds)
  x_train, y_train = zip(*shuffled_ds)

  partitioned_x_train, partitioned_y_train = [partition.fn(data) for data in (x_train, y_train)]

  return (partitioned_x_train, partitioned_y_train), (x_test, y_test)
