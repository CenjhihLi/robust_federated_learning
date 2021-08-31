# -*- coding: utf-8 -*-
"""

"""
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow.keras as keras
#import extra_keras_datasets.emnist as emnist
import nest_asyncio

from client import Client
from server import Server

nest_asyncio.apply()
#num_epoch=10
#batch_size=20
#(train_images, train_labels), (val_images, val_labels) = emnist.load_data(type="letters")
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


def tffdataset2array(dataset):
    xdata=[]
    ydata=[]
    for x , y in dataset.as_numpy_iterator():
        xdata.append(x)
        ydata.append(y)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    ydata.reshape([-1,1])
    return xdata, ydata


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


# Wrap a Keras model for use with TFF.
def model_fn():
    """
    use pretrained ResNet50 here
    without top-fully-connect layer
    """
    input = tf.keras.Input(shape=(32,32,3))
    efnet=tf.keras.applications.resnet50.ResNet50(include_top=False, 
                                weights='imagenet', input_tensor=input)
    gap = tf.keras.layers.GlobalMaxPooling2D()(efnet.output)
    output = tf.keras.layers.Dense(10, activation='softmax', use_bias=True)(gap)
    model = tf.keras.Model(efnet.input, output) 
    return tff.learning.from_keras_model(
        model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


def mlp_model_factory():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])


# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.build_federated_averaging_process(
  model_fn,
  client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
state = trainer.initialize()

# Grab a single batch of data so that TFF knows what data looks like.
#train_data = get_train_data(client = 3, reshape=[28,28],
#                            channel = 3,resize=[32,32],
#                            num_epoch=num_epoch,batch_size=batch_size)
train_data = get_train_data(client = 20)

input_spec = tf.nest.map_structure(
    lambda x: x.element_spec, train_data[0])


model_factory = mlp_model_factory

clients = [
    Client(i, data, model_factory)
    for i, data in enumerate([tffdataset2array(dataset) for dataset in train_data])
    ]



for _ in range(100):
  state, metrics = trainer.next(state, train_data)
  print (metrics['train']['loss'])
  
  
@tff.tf_computation()
def scale(value, factor):
    return tf.nest.map_structure(lambda x: x * factor, value)

@tff.tf_computation()
def unscale(value, factor):
    return tf.nest.map_structure(lambda x: x / factor, value)

@tff.tf_computation()
def add_one(value):
    return value + 1.0

class ExampleTaskFactory(tff.aggregators.UnweightedAggregationFactory):
    
    def __init__(self, inner_factory=None):
        if inner_factory is None:
            inner_factory = tff.aggregators.SumFactory()
        self._inner_factory = inner_factory
            
    def create(self, value_type):
        inner_process = self._inner_factory.create(value_type)
    
        @tff.federated_computation()
        def initialize_fn():
            my_state = tff.federated_value(0.0, tff.SERVER)
            inner_state = inner_process.initialize()
            return tff.federated_zip((my_state, inner_state))
        
        @tff.federated_computation(initialize_fn.type_signature.result,
                                   tff.type_at_clients(value_type))
        def next_fn(state, value):
            my_state, inner_state = state
            my_new_state = tff.federated_map(add_one, my_state)
            my_state_at_clients = tff.federated_broadcast(my_new_state)
            scaled_value = tff.federated_map(scale, (value, my_state_at_clients))
            
            # Delegation to an inner factory, returning values placed at SERVER.
            inner_output = inner_process.next(inner_state, scaled_value)
            
            unscaled_value = tff.federated_map(unscale, (inner_output.result, my_new_state))
            
            new_state = tff.federated_zip((my_new_state, inner_output.state))
            measurements = tff.federated_zip(
                collections.OrderedDict(
                    scaled_value=inner_output.result,
                    example_task=inner_output.measurements))
            
            return tff.templates.MeasuredProcessOutput(
                state=new_state, result=unscaled_value, measurements=measurements)
        
        return tff.templates.AggregationProcess(initialize_fn, next_fn)

client_data = (1.0, 2.0, 5.0)
# Note the inner delegation can be to any UnweightedAggregaionFactory.
# In this case, each factory creates process that multiplies by the iteration
# index (1, 2, 3, ...), thus their combination multiplies by (1, 4, 9, ...).
factory = ExampleTaskFactory(ExampleTaskFactory())
aggregation_process = factory.create(tff.TensorType(tf.float32))
state = aggregation_process.initialize()

output = aggregation_process.next(state, client_data)
print('| Round #1')
print(f'|           Aggregation result: {output.result}   (expected 8.0)')
print(f'| measurements[\'scaled_value\']: {output.measurements["scaled_value"]}')
print(f'| measurements[\'example_task\']: {output.measurements["example_task"]}')

output = aggregation_process.next(output.state, client_data)
print('\n| Round #2')
print(f'|           Aggregation result: {output.result}   (expected 8.0)')
print(f'| measurements[\'scaled_value\']: {output.measurements["scaled_value"]}')
print(f'| measurements[\'example_task\']: {output.measurements["example_task"]}')