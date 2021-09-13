#reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
from dataclasses import dataclass
from typing import Sequence, Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PartitionParams:
  mu: float = 1.5
  sigma: float = 0.45
  k: int = 100
  n: int = 60000
  min_value: int = 0


@dataclass(frozen=True)
class Partition:
  parts: Sequence[int]
  params: PartitionParams
  fn: Callable

  @classmethod
  def _as_fn(cls, parts: Sequence[int], n: int):
    sums = np.cumsum(parts)

    # the sum of all partition can be bigger than our source array
    # because of rounding we find and truncate extra partitions
    index_of_max_sample = np.searchsorted(sums, n, side='right')
    sums = sums[:index_of_max_sample]

    # return partition_sizes[:index_of_max_sample]
    def partition_fn(arr):
      p = np.split(arr[:sums[-1]], sums[:-1])
      p = np.extract([len(c) != 0 for c in p], p)
      return p

    return partition_fn

  @classmethod
  def random_log_normal_partition(cls, params=PartitionParams()):
    mu, sigma, k, n, min_value = vars(params).values()

    alpha = np.random.lognormal(mu, sigma, size=k)
    theta = np.random.dirichlet(alpha)

    parts = np.random.multinomial(n - min_value * k, theta) + min_value
    return Partition(
      parts=parts,
      fn=cls._as_fn(parts, n),
      params=PartitionParams(mu, sigma, k, n, min_value),
    )

  @staticmethod
  def hist(parts, bins=100):
    from matplotlib.ticker import PercentFormatter

    k = len(parts)
    fig = plt.figure(figsize=(5, 3))

    plt.rc('axes', labelsize=10)  # fontsize of the x and y labels

    ax = plt.gca()
    ax.set_xlabel('Client sample size')
    ax.set_ylabel('Frequency ')

    ax.yaxis.set_major_formatter(PercentFormatter(decimals=1))  # FuncFormatter('{0:.0%}'.format))

    plt.hist(parts, bins=bins, weights=np.repeat(1 / k, k))
    plt.tight_layout()
    fig.savefig('partition.pdf', format='pdf')
    plt.show()
