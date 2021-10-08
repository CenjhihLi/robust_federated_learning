# A robust federated learning (robust optimization)
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

we using some useful function refer from their code (but change quite much) and apply our gamma mean as an aggregator

Author: Cen-Jhih Li
Belongs: Academic Senica, Institute of statistic, Robust federated learning project
# Dataset: MNIST, EMNIST

Contains the emnist, mnist, fashion_MNIST, chest xray data. 

# Build environment in anaconda
see build_env.txt 

# Build environment via Docker:
docker "args" pull nvcr.io/nvidia/tensorflow:21.09-tf2-py3
pip install matplotlib wquantiles nest_asyncio 

Ubundo 16, 18, and 20 should all be fine

if using EMNIST, then install tensorflow-federated

# Training

Using the command in cmd_run.txt to implement the experiments.

# Results

After the experiments are done, execute `mnist_results_plots.ipynb` using Jupyter notebook.
