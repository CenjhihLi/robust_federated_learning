#import sys
#sys.path.append("C:/GitHub\robust_federated_learning")
import tensorflow.keras as keras
import util.experiment_runner as experiment_runner
import tensorflow as tf
import gc

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

def LeNet_model_factory():
    return keras.models.Sequential([
        keras.layers.Conv2D(filters=20, kernel_size=5, 
                            input_shape=(28, 28, 1), 
                            activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, strides = 2, padding='same'),
        keras.layers.Conv2D(filters=50, kernel_size=5, 
                            activation='relu', 
                            padding='same'),
        keras.layers.MaxPool2D(pool_size=2, strides = 2, padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10, activation='softmax')
        ])
#seems not better
###################################
#####          usage          #####
#model_factory = cnn_model_factory#
#     input_shape = [28,28,1]     #
#model_factory = mlp_model_factory#
#        input_shape =[-1]        #
###################################
def res_model(pooling = None):
    def res_model_factory():
        from tensorflow.keras.applications import resnet50
        model=resnet50.ResNet50(input_shape=(150,150,3), weights='imagenet', include_top=False, pooling = pooling)
        x = model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.7, name='dropout1')(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.5, name='dropout2')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3, name='dropout3')(x)
        x = keras.layers.Dense(1,activation='sigmoid')(x)
        for layer in model.layers:
            layer.trainable = False
        net_final = keras.Model(inputs=model.input, outputs=x)
        return net_final
    return res_model_factory

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
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization(name='bn4')(x)
    x = keras.layers.SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), name='pool4')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.7, name='dropout1')(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5, name='dropout2')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3, name='dropout3')(x)
    x = keras.layers.Dense(1,activation='sigmoid')(x)
    
    model = keras.Model(inputs=input_img, outputs=x)
    return model

def CNN_model_xray_initialize(model):
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