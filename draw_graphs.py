import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-paper')

def plot_range(experiments, ax, plot_start, plot_end, metric_idx=1, ylim=None):
#     if metric_idx == 1:
#         ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
#         ax.set_ylabel('Accuracy')
#     else:
#         ax.set_ylabel('Loss')
        
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    for path, label, alpha, linestyle in experiments:
#         if alpha == 1:
#             marker = 'P'
#         elif alpha == 0.8:
#             marker = '^'
#         else:
#             marker = 'd'
            
        if Path(f'{path}.npz').is_file():
            history = np.load(f'{path}.npz')['history']

            ax.plot(range(plot_start, min(len(history), plot_end)),
                    history[plot_start:min(len(history), plot_end), metric_idx], label=label, #alpha=alpha, #color=color,# marker=marker,
                    linestyle=linestyle)

    if ylim is not None:
        ax.set_ylim(ylim)
#     ax.set_xlabel('Round')


def plot(experiments, metric_idx=1):
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots()

    plot_range(experiments, ax1, 0, 800, metric_idx=metric_idx)  # , ylim=(0.92, 0.95))
    # plot_range(experiments, ax2, 500, 570, ylim=(0.6, 0.99))

    # ax2.legend(loc='lower right')  # , title=title)


def plot_all(experiment, seed, cpr, attack_type, real_alpha):
    real_alpha = int(real_alpha * 100)
    
    plot([
        (f'experiments/{experiment}/seed_{seed}/mean_cpr_{cpr}', 'mean[1]', '#FFB3B3', '-'),
        (f'experiments/{experiment}/seed_{seed}/mean_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 'mean[1] byz', '#FFB3B3', ':'),

        (f'experiments/{experiment}/seed_{seed}/mean_t_30_cpr_{cpr}', 'mean[U]', '#FF3333', '-'),
        (f'experiments/{experiment}/seed_{seed}/mean_t_30_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 'mean[U] byz', '#FF3333', ':'),

        (f'experiments/{experiment}/seed_{seed}/mean_w_cpr_{cpr}', 'mean[W]', '#B30000', '-'),
        (f'experiments/{experiment}/seed_{seed}/mean_w_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 'mean[W] byz', '#B30000', ':'),

        (f'experiments/{experiment}/seed_{seed}/median_cpr_{cpr}', 'median[1]', '#d9e5ff', '-'),
        (f'experiments/{experiment}/seed_{seed}/median_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 'median[1] byz', '#d9e5ff', ':'),

        (f'experiments/{experiment}/seed_{seed}/median_t_30_cpr_{cpr}', 'median[U]', '#3377FF', '-'),
        (f'experiments/{experiment}/seed_{seed}/median_t_30_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 'median[U] byz', '#3377FF',':'),

        (f'experiments/{experiment}/seed_{seed}/median_w_cpr_{cpr}', 'median[W]', '#003CB3', '-'),
        (f'experiments/{experiment}/seed_{seed}/median_w_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 'median[W] byz', '#003CB3', ':'),
        
        (f'experiments/{experiment}/seed_{seed}/t_mean_20_cpr_{cpr}', 't_mean[1]', '#00FF00', '-'),
        (f'experiments/{experiment}/seed_{seed}/t_mean_20_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 't_mean[1] byz', '#00FF00', ':'),

        (f'experiments/{experiment}/seed_{seed}/t_mean_20_t_30_cpr_{cpr}', 't_mean[U]', '#009900', '-'),
        (f'experiments/{experiment}/seed_{seed}/t_mean_20_t_30_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 't_mean[U] byz', '#009900',':'),

        (f'experiments/{experiment}/seed_{seed}/t_mean_20_w_cpr_{cpr}', 't_mean[W]', '#006600', '-'),
        (f'experiments/{experiment}/seed_{seed}/t_mean_20_w_cpr_{cpr}_b_{attack_type}_{real_alpha}_100000', 't_mean[W] byz', '#006600', ':'),
    ])
    
    plt.legend()
    
    plt.show()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw some graphs.')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('seed', type=int, help='seed number')
    parser.add_argument('cpr', type=str, help='clients per round')
    parser.add_argument('attack_type', type=str, help='attack_type')
    parser.add_argument('real_alpha', type=float, help='real alpha')

    args = parser.parse_args()

    experiment, seed, cpr, attack_type, real_alpha = vars(args).values()
    
    plot_all(experiment, seed, cpr, attack_type, real_alpha)
