import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from prepare_data.transform import tfdataset2array
# Load simulation data.
train_emnist, test_emnist = tff.simulation.datasets.emnist.load_data()

# Pick a subset of client devices to participate in training.
def get_train_data(client = len(train_emnist.client_ids), 
                   reshape=[-1]):
    def client_data(n, reshape = [-1]):
        client_data = train_emnist.create_tf_dataset_for_client(train_emnist.client_ids[n])
        client_data = client_data.map(
            lambda e: (tf.reshape(e['pixels'], reshape)/255 ,
                       e['label'])
            )  
        return client_data
    return [client_data(n,reshape=reshape) for n in range(client)]


def get_test_data(client = len(test_emnist.client_ids), 
                   reshape=[-1]):
    def client_data(n, reshape = [-1]):
        client_data = test_emnist.create_tf_dataset_for_client(test_emnist.client_ids[n])
        client_data = client_data.map(
            lambda e: (tf.reshape(e['pixels'], reshape)/255 ,
                       e['label'])
            )  
        return client_data
    return [client_data(n,reshape=reshape) for n in range(client)]

def load(client, reshape = [-1]):
    """
    training data shape: (#observations, #pixels=784)
    testing data shape: (#observations, #pixels=784)
    if we change the model into CNN, reshape([-1,28,28,1])
    """
    train_data = get_train_data(client = client, reshape=reshape)
    train_data = [tfdataset2array(dataset) for dataset in train_data]
  
    test = get_test_data(client = client, reshape=reshape)
    test = [tfdataset2array(dataset) for dataset in test]
    test_x, test_y = zip(*test)
    return train_data, test_x, test_y
