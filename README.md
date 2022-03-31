# A robust federated learning
Some experiment code refer from https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

The server aggregator is `numpy` code, which computes by cpu, and we find that the computation seems large.
However, cupy and tf.tensor is not as convinient as numpy. 

The aggregator can be easily applied to Nvidia's NVFlare and Clara train which are developed for real federated learning system.

Author: Cen-Jhih Li  
Belongs: Academia Sinica, Institute of Statistical Science, Robust federated learning project
# Datasets: 

Including emnist, mnist, fashion_MNIST, chest xray data. 
# Build environment
see `build_env.txt`

Ubundo 16, 18, and 20 should all be fine

remove the EMNIST part in the code and no need to install tensorflow-federated if not using EMNIST
# Training

Using the command in `cmd_run.txt` to implement the experiments. 
# Python scripts

main script (run experiments): `run_experiment.py`  
experiment functions (include loading previous training results) are in: `./util/experiment_runner.py`  
server class (include server training procedure) defined in: `./util/server.py`  
client class (include client training procedure) defined in: `./util/client.py`  
neural network model defined in: `./util/model.py`  
datasets functions in folder: `./prepare_data/`  
(Pneumonia dataset provided by: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 not provide in this github project)

# Results

After the experiments are done, execute `mnist_results_plots.ipynb` using Jupyter notebook.
Our simulation and experiments results are in `aggregators_simulation.ipynb` and `mnist_results_plots.ipynb` 
