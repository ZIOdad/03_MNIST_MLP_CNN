# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
print(__file__)
print(os.getcwd())
print(os.path.dirname(os.path.realpath(__file__)))

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())

def makedir(filepath):
    try:
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
            print(f"MAKE DIR at {filepath}")
        else:
            print(f"AREADY EXIST DIR at {filepath}")
    except OSError:
        print("FAILED TO CREATE DIR")

import tensorflow as tf

def fixerr():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
fixerr()

# Simple CNN for the MNIST Dataset
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *

import keras_tuner as kt

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
#(60000, 28, 28, 1) (10000, 28, 28, 1) 흑백이므로 1(0~255)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# define a simple CNN model
def baseline_model(hp):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(
                    hp.Choice('units',[8, 16, 32]),
                    activation='relu'))
    model.add(Dense(num_classes, kernel_regularizer='l2', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# hyper parameter optimization
tuner = kt.RandomSearch(baseline_model, objective='val_loss', max_trials=5)
tuner.search(X_train, y_train, epochs=5, batch_size=200, validation_split=0.2, verbose=1)
best_model = tuner.get_best_models(num_models=3)[0]
loss, err = best_model.evaluate(X_test, y_test)
