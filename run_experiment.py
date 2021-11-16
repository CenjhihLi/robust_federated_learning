#import sys
#sys.path.append("C:/GitHub\robust_federated_learning")
import tensorflow.keras as keras
import util.experiment_runner as experiment_runner
import numpy as np
import tensorflow as tf
import random
import os
import gc
from tensorflow.keras.applications import resnet50, xception

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
seed=1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

we using some useful function refer from their code (but change quite a lot) and apply our gamma mean and geometric median as aggregators
"""

def mlp_model_factory():
    return keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
        ])
#model_factory = mlp_model_factory

def cnn_model_factory():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, 
                            input_shape=(28, 28, 1), activation='relu', 
                            padding='same'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
        ])

def cnn2_model_factory():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, 
                            input_shape=(28, 28, 1), activation='relu', 
                            padding='same'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(filters=64, kernel_size=3, 
                            activation='relu', 
                            padding='same'),
        keras.layers.MaxPool2D(pool_size=2),
        keras.layers.Conv2D(filters=128, kernel_size=3, 
                            activation='relu', 
                            padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
        ])
#seems not better

#model_factory = cnn_model_factory
#input_shape = [28,28,1]
#model_factory = mlp_model_factory
#input_shape = [-1]

def res_model_factory():
    model=resnet50.ResNet50(input_shape=(150,150,3), weights='imagenet', include_top=False, )
    x = model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256,activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1,activation='sigmoid')(x)
    for layer in model.layers:
        layer.trainable = False
    net_final = keras.Model(inputs=model.input, outputs=x)
    return net_final

def CNN_model_xray():
    input_img = keras.layers.Input(shape=(224,224,3), name='ImageInput')
    x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
    x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), name='pool1')(x)
    
    x = keras.layers.SeparableConv2D(128, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.SeparableConv2D(128, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), name='pool2')(x)
    
    x = keras.layers.SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), name='pool3')(x)
    
    x = keras.layers.SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization(name='bn4')(x)
    x = keras.layers.SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), name='pool4')(x)
    
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.7, name='dropout1')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5, name='dropout2')(x)
    x = keras.layers.Dense(1,activation='sigmoid')(x)
    
    model = keras.Model(inputs=input_img, outputs=x)
    return model

if __name__ == '__main__':
    clients=20
    model = mlp_model_factory
    input_shape = [-1]
    experiment_runner.run_all('expr_no_attacks',
                              model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0,
                              t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    experiment_runner.run_all('expr_random',
                              model, input_shape=input_shape, dataset='mnist', 
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                              num_samples_per_attacker=1_000_000, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    model = cnn_model_factory
    input_shape=[28,28,1]
    experiment_runner.run_all('cnn_expr_no_attacks',
                              model_factory = model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, 
                              real_alpha=0, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    experiment_runner.run_all('cnn_expr_random', 
                              model_factory = model, input_shape=input_shape, dataset='mnist',
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                              num_samples_per_attacker=1_000_000, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    experiment_runner.run_all('fashion_mnist_cnn_expr_no_attacks',
                              model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, 
                              real_alpha=0, t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    experiment_runner.run_all('fashion_mnist_cnn_expr_random', 
                              model_factory = model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                              num_samples_per_attacker=1_000_000, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    model = mlp_model_factory
    input_shape = [-1]
    experiment_runner.run_all('fashion_mnist_expr_no_attacks',
                              model, input_shape=input_shape, dataset='fashion_mnist',
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0,
                              t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    experiment_runner.run_all('fashion_mnist_expr_random',
                              model, input_shape=input_shape, dataset='fashion_mnist', 
                              seed=seed, cpr='all', rounds=1000, mu=1.5, sigma=3.45, real_alpha=0.1,
                              num_samples_per_attacker=1_000_000, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7)

    clients=3
    #model = res_model_factory()
    #input_shape = [150,150,3]
    #initialize = None

    model = CNN_model_xray
    input_shape = [224,224,3]
    def initialize(model):
        from tensorflow.keras.applications import vgg16
        VGG16_model = vgg16.VGG16(input_shape=(224,224,3), weights='imagenet', include_top=True, )
        f = VGG16_model.get_weights()
        w,b = f[0], f[1]
        model.layers[1].set_weights = [w,b]
        w,b = f[2], f[3]
        model.layers[2].set_weights = [w,b]
        w,b = f[4], f[5]
        model.layers[4].set_weights = [w,b]
        w,b = f[6], f[7]
        model.layers[5].set_weights = [w,b]
        del VGG16_model, f, w, b
        gc.collect()
        return model

    experiment_runner.run_all('pneumonia_expr_no_attacks',
                              model, input_shape=input_shape, dataset='pneumonia',
                              #CNN_model_xray, input_shape=[224,224,3], dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, mu=1.5, sigma=3.45, real_alpha=0,
                              t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7, initialize = initialize)

    experiment_runner.run_all('pneumonia_expr_random',
                              model, input_shape=input_shape, dataset='pneumonia', 
                              #CNN_model_xray, input_shape=[224,224,3], dataset='pneumonia',
                              seed=seed, cpr='all', rounds=100, mu=1.5, sigma=3.45, real_alpha=0.1,
                              num_samples_per_attacker=1_000_000, 
                              attack_type='random', t_mean_beta=0.1, clients=clients,
                              gam_max=10, gamma=0.5, geo_max=1000, tol = 1e-7, initialize = initialize)

