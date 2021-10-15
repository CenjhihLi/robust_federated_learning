# A robust federated learning (robust optimization)
Some experiment code refer from https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

We consider the sample size declare from sample size originally, but remove it since it is not the major problem in our method


The server aggregator using numpy, but we find that the computation seems large
However, cupy and tf.tensor is not as convinient as numpy 

The aggregator can be easily applied to Nvidia's NVFlare and Clara train

Author: Cen-Jhih Li
Belongs: Academia Sinica, Institute of Statistical Science, Robust federated learning project
# Dataset: MNIST, EMNIST

Contains the emnist, mnist, fashion_MNIST, chest xray data. 

# Build environment in anaconda
see build_env.txt 

# Build environment via Docker:
docker "args" pull nvcr.io/nvidia/tensorflow:21.09-tf2-py3
pip install matplotlib wquantiles nest_asyncio cupy-cuda101 tensorflow-federated==0.17.0

using medMNIST:
python -m pip install -U setuptools pip
pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git


Ubundo 16, 18, and 20 should all be fine

if don't using EMNIST, remove the EMNIST import in the code and do not need to install tensorflow-federated

# Training

Using the command in cmd_run.txt to implement the experiments.

# Results

After the experiments are done, execute `mnist_results_plots.ipynb` using Jupyter notebook.
Our simulation and experiments results are in `aggregators_simulation.ipynb` and `mnist_results_plots.ipynb` 
