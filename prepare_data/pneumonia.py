import random
import gc
import cv2
import tensorflow as tf
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from prepare_data.transform import tfdataset2array
from prepare_data.partition_v2 import Partition, PartitionParams
"""
Use pneumonia dataset provided by:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""
#datagen = ImageDataGenerator(rescale=1./255)
#train_generator = datagen.flow_from_directory(
#        './prepare_data/CellData/chest_xray/train',
#        target_size=(150, 150),
#        batch_size=32,
#        class_mode='binary')
def load_data(path):
    import pandas as pd
    normal_cases_dir = path / 'NORMAL'
    pneumonia_cases_dir = path / 'PNEUMONIA'

    # Get the list of all the images
    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

    # An empty list. We will insert the data into this list in (img_path, label) format
    data = []

    # Go through all the normal cases. The label for these cases will be 0
    for img in normal_cases:
        data.append((img,0))

    # Go through all the pneumonia cases. The label for these cases will be 1
    for img in pneumonia_cases:
        data.append((img, 1))

    # Get a pandas dataframe from the data we have in our list 
    data = pd.DataFrame(data, columns=['image', 'label'],index=None)

    # Shuffle the data 
    data = data.sample(frac=1.).reset_index(drop=True)
    return data
    
def load_img(path, input_shape = (224,224)):
    normal_cases_dir = path / 'NORMAL'
    pneumonia_cases_dir = path / 'PNEUMONIA'

    normal_cases = normal_cases_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

    img_data = []
    img_labels = []
    
    # Normal cases
    for img in normal_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, input_shape)
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = 0.
        img_data.append(img)
        img_labels.append(label)
                      
    # Pneumonia cases        
    for img in pneumonia_cases:
        img = cv2.imread(str(img))
        img = cv2.resize(img, input_shape)
        if img.shape[2] ==1:
            img = np.dstack([img, img, img])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        label = 1.
        img_data.append(img)
        img_labels.append(label)
    
    return np.array(img_data).astype(np.float32), np.array(img_labels).astype(np.float32)

def load(partition_config, input_shape = [150,150,3], load_train_from_dir = False, self_split_val: bool=False):
    """
    training data shape: (#observations, #pixels=[150,150,3])
    testing data shape: (#observations, #pixels=[150,150,3])
    reshape([-1,150,150,3])
    """
    from pathlib import Path
    root_path = Path('./prepare_data/CellData/chest_xray')
    train_dir = root_path / 'train'
    val_dir = root_path / 'val'
    test_dir = root_path / 'test'

    train = image_dataset_from_directory(
            train_dir,
            label_mode='binary',
            batch_size=5232,
            image_size=(input_shape[0], input_shape[1]))

    x_train, y_train = tfdataset2array(train)
    x_train = np.divide(tf.reshape(x_train[0], [x_train[0].shape[0]] + input_shape), 255., dtype=np.float32)
    y_train = y_train.squeeze(0).astype(np.float32).reshape( (-1,) )
    #y_train = tf.reshape(y_train ,[-1,1])

    val = image_dataset_from_directory(
            val_dir,
            label_mode='binary',
            batch_size=16,
            image_size=(input_shape[0], input_shape[1]))

    val_x, val_y = tfdataset2array(val)
    val_x = np.divide(tf.reshape(val_x[0], [val_x[0].shape[0]] + input_shape), 255., dtype=np.float32)
    val_y = val_y.squeeze(0).astype(np.float32).reshape( (-1,) )
    #val_y = tf.reshape(val_y,[-1,1])

    if self_split_val:
        x_train = np.concatenate((x_train, val_x), axis = 0)
        y_train = np.concatenate((y_train, val_y), axis = 0)
        shuffled = np.random.permutation(5232)
        val_x = x_train[shuffled,...][:int(5232*0.1):,...]
        val_y = y_train[shuffled,...][:int(5232*0.1):,...]
        x_train = x_train[shuffled,...][int(5232*0.1):,...]
        y_train = y_train[shuffled,...][int(5232*0.1):,...]
        del shuffled
        gc.collect()

    partition = Partition.random_log_normal_partition(
    PartitionParams(
            mu=partition_config['mu'],
            sigma=partition_config['sigma'],
            k=partition_config['#clients'],
            n=x_train.shape[0],
            min_value = partition_config['min_value'],
            ))

    shuffled_ds = list(zip(x_train, y_train))
    random.shuffle(shuffled_ds)
    x_train, y_train = zip(*shuffled_ds)

    partitioned_x_train, partitioned_y_train = [partition.fn(data) for data in (x_train, y_train)]

    test = image_dataset_from_directory(
            test_dir,
            label_mode='binary',
            batch_size=624,
            image_size=(input_shape[0], input_shape[1]))

    test_x, test_y = tfdataset2array(test)
    test_x = np.divide(tf.reshape(test_x[0], [test_x[0].shape[0]] + input_shape), 255., dtype=np.float32)
    test_y = test_y.squeeze(0).astype(np.float32).reshape( (-1,) )
    #test_y = tf.reshape(test_y,[-1,1])

    train_data = zip(partitioned_x_train, partitioned_y_train)
    del partitioned_x_train, partitioned_y_train, test, x_train, y_train, shuffled_ds, train
    gc.collect()
    return train_data, (val_x, val_y), (test_x, test_y)
