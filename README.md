# A robust federated learning (robust optimization)
Some experiment code refer from https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

The server aggregator is numpy code, but we find that the computation seems large
However, cupy and tf.tensor is not as convinient as numpy 

The aggregator can be easily applied to Nvidia's NVFlare and Clara train

Author: Cen-Jhih Li
Belongs: Academia Sinica, Institute of Statistical Science, Robust federated learning project
# Dataset: 

Including emnist, mnist, fashion_MNIST, chest xray data. 

# Build environment in anaconda
see build_env.txt 

Ubundo 16, 18, and 20 should all be fine

remove the EMNIST part in the code and do not need to install tensorflow-federated if not use EMNIST
# Training

Using the command in cmd_run.txt to implement the experiments. 
(with conda env cmds. if using docker, ignore the conda parts)
# Results

After the experiments are done, execute `mnist_results_plots.ipynb` using Jupyter notebook.
Our simulation and experiments results are in `aggregators_simulation.ipynb` and `mnist_results_plots.ipynb` 
