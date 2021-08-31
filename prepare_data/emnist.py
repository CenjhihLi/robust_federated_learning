"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting
functions: fs_setup,

Author: Cen-Jhih Li
Belongs: Academic Senica, Institute of statistic, Robust federated learning project
"""
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

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


def tffdataset2array(dataset):
    xdata=[]
    ydata=[]
    for x, y in dataset.as_numpy_iterator():
        xdata.append(x)
        ydata.append(y)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    ydata.reshape([-1,1])
    return xdata, ydata
"""
training data shape: (#observations, #pixels=784)
testing data shape: (#observations, #pixels=784)
if we change the model into CNN, reshape([-1,28,28,1])
"""
def prepare_data(client, reshape = [-1]):
    train_data = get_train_data(client = client, reshape=reshape)
    train_data = [tffdataset2array(dataset) for dataset in train_data]
  
    test = get_test_data(client = client, reshape=reshape)
    test = [tffdataset2array(dataset) for dataset in test]
    test_x, test_y = zip(*test)
    return train_data, test_x, test_y

## Pick a subset of client devices to participate in training.
#def get_train_data(client = len(train_emnist.client_ids), 
#                   reshape=[-1], channel = 1, resize=[28,28],
#                   num_epoch=10,batch_size=20):
#    def client_data(n, reshape = [-1], channel = 1,resize=[28,28],num_epoch=10,batch_size=20):
#        client_data = train_emnist.create_tf_dataset_for_client(train_emnist.client_ids[n])
#        client_data = client_data.map(
#            lambda e: (tf.image.resize(tf.repeat(tf.expand_dims(tf.reshape(
#                e['pixels'], reshape), axis = -1),3,axis = -1)/255, resize) ,
#                e['label'])
#            ).repeat(num_epoch).batch(batch_size)  
#        return client_data
#    return [client_data(n,reshape=reshape,channel=channel, resize=resize,
#                        num_epoch=num_epoch,batch_size=batch_size) for n in range(client)]


#Grab a single batch of data so that TFF knows what data looks like.
#train_data = get_train_data(client = 3, reshape=[28,28],
#                            channel = 3,resize=[32,32],
#                            num_epoch=num_epoch,batch_size=batch_size)


