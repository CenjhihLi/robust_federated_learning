"""
reference: https://github.com/amitport/Towards-Federated-Learning-with-Byzantine-Robust-Client-Weighting

add random attack
"""

import numpy as np
import tensorflow as tf
from prepare_data.transform import tfdataset2array
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Client():
  def __init__(self, idx, data, model_factory, epochs):
    self.idx = idx

    self.attacker = False
    self.threat_model = None

    self.num_of_samples = len(data[0])

    self._x, self._y = data[0], data[1]

    self._model = model_factory()
    self._epochs = epochs

  def as_attacker(self, threat_model):
    self.attacker = True
    self.threat_model = threat_model
    
    if self.threat_model.type == 'y_flip':
      self._y = 9 - self._y #not use
    
    self.num_of_samples = self.threat_model.num_samples_per_attacker

  def train(self, server_weights, lr_decayed, optimizer, loss_fn, metrics, val_x, val_y, x_chest: bool, chkpt_path):   
    if self.attacker and self.threat_model is not None and self.threat_model.type == 'delta_to_zero':
      return_deltas = list()
      for i, layer in enumerate(self._model.layers):
        weights = layer.get_weights()
        for weight in weights:
          if layer.trainable:
            return_deltas.append( -weight )
          else: 
            return_deltas.append( np.zeros(shape = weight.shape, dtype = np.float32) )
      return return_deltas #not use
      
    if self.attacker and self.threat_model is not None and self.threat_model.type == 'random':
      return_deltas = list()
      for _, layer in enumerate(self._model.layers):
        weights = layer.get_weights()
        for weight in weights:
          if layer.trainable:
            return_deltas.append( np.random.normal(loc=5.0, scale=1.0, size = weight.shape) )
          else: 
            return_deltas.append( np.zeros(shape = weight.shape, dtype = np.float32) )
      return return_deltas

    self._model.set_weights(server_weights)
    if x_chest:
      self._model.compile(
        #optimizer = tf.keras.optimizers.Adam( learning_rate = lr_decayed ), 
        optimizer = optimizer( learning_rate = lr_decayed, decay = 1e-5),
        loss = loss_fn,
        metrics = metrics,
        #metrics = [
        #    tf.keras.metrics.Precision(name='precision'),
        #    tf.keras.metrics.Recall(name='recall')
        #],
      )
      #datagen = ImageDataGenerator(
      #  rescale=1./255,
      #  horizontal_flip=True,
      #  rotation_range=0.2,
      #  brightness_range=(1.2,1.5),
      #  )
      #train_generator = datagen.flow((self._x, self._y),
      #  batch_size=16,
      #  )
      es = EarlyStopping(patience=5)
      chkpt_path = chkpt_path / f'client{self.idx}'
      chkpt_path.mkdir(parents=True, exist_ok=True)
      chkpt = ModelCheckpoint(filepath=chkpt_path/'model', save_best_only=True, save_weights_only=True)
      self._model.fit(
        #train_generator,
        self._x, self._y,
        batch_size = 16,
        steps_per_epoch = np.maximum(self.num_of_samples//16, 1),
        epochs = self._epochs,
        validation_data =(val_x, val_y),
        callbacks = [es, chkpt],
        class_weight = {0:1.0, 1:0.35},
        )
    else:
      #Since local machine do not have last update v and only iterate once, Adam is not work here, should employ Adam in server
      self._model.compile(
          #optimizer = tf.keras.optimizers.Adam( learning_rate = lr_decayed ), 
          optimizer = optimizer( learning_rate = lr_decayed ),
          loss = loss_fn,
          metrics = metrics,
      )
      self._model.fit(self._x, self._y, verbose = 0,
                      epochs = self._epochs, steps_per_epoch = 1,
                      # go over 10% of data like in Yin's paper
                      batch_size = max((self.num_of_samples // 10), 1), 
                      # epochs=3, batch_size=50,
                      #                         callbacks=[tf.keras.callbacks.EarlyStopping(
                      #                             monitor='loss', patience=1, restore_best_weights=True)]
                      )

    new_weights = self._model.get_weights()

    delta_weights = [new_w - old_w for new_w, old_w in zip(new_weights, server_weights)]

    if self.attacker and self.threat_model is not None and self.threat_model.type == 'sign_flip':
      return [-t for t in delta_weights] #not use
    else:
      return delta_weights
