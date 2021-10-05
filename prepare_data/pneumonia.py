import random
import tensorflow as tf
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from prepare_data.transform import tfdataset2array
from prepare_data.partition_v2 import Partition, PartitionParams

#datagen = ImageDataGenerator(rescale=1./255)
#train_generator = datagen.flow_from_directory(
#        './prepare_data/CellData/chest_xray/train',
#        target_size=(150, 150),
#        batch_size=32,
#        class_mode='binary')

def load(partition_config, input_shape = [150,150,3]):
    """
    training data shape: (#observations, #pixels=[150,150,3])
    testing data shape: (#observations, #pixels=[150,150,3])
    reshape([-1,150,150,3])
    """
    train = image_dataset_from_directory(
            './prepare_data/CellData/chest_xray/train',
            label_mode='categorical',
            batch_size=5232,
            image_size=(150, 150))

    x_train, y_train = tfdataset2array(train)
    x_train = np.divide(tf.reshape(x_train[0], [x_train[0].shape[0]] + input_shape), 255., dtype=np.float32)
    y_train = y_train[0]

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

    test = image_dataset_from_directory(
            './prepare_data/CellData/chest_xray/test',
            label_mode='categorical',
            batch_size=624,
            image_size=(150, 150))

    test_x, test_y = tfdataset2array(test)
    test_x = np.divide(tf.reshape(test_x[0], [test_x[0].shape[0]] + input_shape), 255., dtype=np.float32)
    test_y = test_y[0]

    train_data = zip(partitioned_x_train, partitioned_y_train)
    return train_data, (test_x, test_y)
