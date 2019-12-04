# MobileNet 

import tensorflow as tf
import json

import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from tensorflow.python.client import device_lib
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def load_data(name):
    data, ds_info = tfds.load(name, with_info=True)
    (train_data, test_data) = data["train"], data["test"]

    num_train = ds_info.splits['train'].num_examples
    num_test = ds_info.splits['test'].num_examples
    print("The number of training sample is", num_train)
    print("The number of testing sample is", num_test)
    return train_data, test_data

def format_data(image, label, num_class, size=224):
    image = tf.dtypes.cast(image, tf.float32)
    image /= 225.0
    image = tf.image.resize(image, size=(size, size))
    label = tf.one_hot(label, depth=num_class)
    return image, label

def _conv_block(x, filters, kernel_size, stride, input_shape, padding='same', alpha=1):

    filters = int(filters * alpha)
    x = tf.keras.layers.Conv2D(filters=filters, 
                                kernel_size=kernel_size,
                                strides=stride,
                                input_shape=input_shape,
                                padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)

    return x

def _deepwise_conv_block(x, filters, kernel_size, stride, padding='same', alpha=1):

    filters = int(filters * alpha)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                        strides=stride,
                                        padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x                                    

def mobilenet(alpha=1, size=224, num_class=10):
    '''
    alpha = [0.25, 0.50, 0.75, 1.0] # width
    size = [128, 160, 192, 224] # resolution 
    '''
    # define input tensor size
    input_shape = (size, size, 3)
    img_input = tf.keras.layers.Input(shape=input_shape)

    x = _conv_block(img_input, filters=32, kernel_size=(3, 3), stride=2, 
                    input_shape=input_shape, alpha=alpha)
    x = _deepwise_conv_block(x, filters=64, kernel_size=(3, 3), stride=1)
    x = _deepwise_conv_block(x, filters=128, kernel_size=(3, 3), stride=2)
    x = _deepwise_conv_block(x, filters=128, kernel_size=(3, 3), stride=1)
    x = _deepwise_conv_block(x, filters=256, kernel_size=(3, 3), stride=2)
    x = _deepwise_conv_block(x, filters=256, kernel_size=(3, 3), stride=1)
    x = _deepwise_conv_block(x, filters=512, kernel_size=(3, 3), stride=2)
    # 5x
    # x = _deepwise_conv_block(x, filters=512, kernel_size=(3, 3), stride=1)
    # x = _deepwise_conv_block(x, filters=512, kernel_size=(3, 3), stride=1)
    # x = _deepwise_conv_block(x, filters=512, kernel_size=(3, 3), stride=1)
    # x = _deepwise_conv_block(x, filters=512, kernel_size=(3, 3), stride=1)
    # x = _deepwise_conv_block(x, filters=512, kernel_size=(3, 3), stride=1)
    # paper parameter mistakes here
    x = _deepwise_conv_block(x, filters=1024, kernel_size=(3, 3), stride=2)
    x = _deepwise_conv_block(x, filters=1024, kernel_size=(3, 3), stride=1)
    # flatten
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(num_class, activation='softmax')(x)
    # define model 
    model = tf.keras.models.Model(inputs=img_input, outputs=output)
    return model

'''
def mobilenet_naive(alpha=1, size=224, num_class=10):
    tf.keras.backend.clear_session()
    # MobileNet 
    model = tf.keras.Sequential()
    # model structure 
    model.add(tf.keras.layers.Conv2D(filters=int(32*alpha), kernel_size=(3,3), strides=2, input_shape=(size, size, 3), padding='same'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(64*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(128*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(128*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(512*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    # 5x
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(512*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(512*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(512*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(512*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=int(512*alpha), kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    # 5x ends
    # here I changed the strides to 1 
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=1, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(activation='relu'))
    # ??? pool_size wrong
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(7,7)))
    # ???
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=num_class, activation='softmax'))
    return model

'''

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        time_pass = np.round(time.time() - self.epoch_time_start, 3)
        self.times.append(time_pass)
        logs['time'] = time_pass


if __name__ == '__main__':
    '''
    alpha = [0.25, 0.50, 0.75, 1.0] # width
    size = [128, 160, 192, 224] # resolution 
    '''

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())

    print("TF version: ",tf.__version__)
    print("Keras version:",tf.keras.__version__)
    
    print(device_lib.list_local_devices())

    

    # preprocess 
    # -----set parameters------
    num_class = 10
    alpha_list = [0.50, 0.75, 1.0]
    size = 224
    rho = np.round(size / 224.0, 2)
    for alpha in [1]:
        train_data, test_data = load_data('cifar10')
        # token = 'cifar100_mn_alpha' + str(alpha) + '_rho' + str(rho) 
        token = 'cifar10_mn_alpha_1_shallow'
        # --------------------------
        train_data = train_data.map(lambda x: format_data(x['image'], x['label'], num_class, size=size))
        test_data = test_data.map(lambda x: format_data(x['image'], x['label'], num_class, size=size))
        # create model  
        model = mobilenet(alpha=alpha, size=size, num_class=num_class)
        # print(model.summary())
        # write model structure 
        with open('models/' + token + '.json', 'w') as f:
            json.dump(model.to_json(), f)

        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        # compile model
        rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.005)
        # adam = tf.keras.optimizers.Adam(learning_rate=0.05)
        training_history = model.compile(optimizer=rmsprop,
                                            metrics=['accuracy'],
                                            loss='categorical_crossentropy')

        filename = "logs/history100/" + token + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'
        csv_logger = tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)
        time_callback = TimeHistory()

        # fit model 
        history = model.fit(train_data.shuffle(1000).batch(64),
                            epochs=10,
                            callbacks=[time_callback, csv_logger],
                            validation_data=test_data.shuffle(1000).batch(64))

        # save
        model.save('models/'+ token + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")+'.h5')


