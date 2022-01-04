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
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *

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
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(96, (6, 6), strides=2, padding='valid', input_shape=(28, 28, 1), activation='relu'))
    #input(mini_batch_size, 28, 28, 1) ==> output(mini_batch_size, 12, 12, 96)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    #input(mini_batch_size, 12, 12, 96) ==> output(mini_batch_size, 6, 6, 96)
    model.add(BatchNormalization())
    model.add(Conv2D(256, (6, 6), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(BatchNormalization())
    model.add(Conv2D(384, (6, 6), padding='same', activation='relu'))
    model.add(Conv2D(384, (6, 6), padding='same', activation='relu'))
    model.add(Conv2D(256, (6, 6), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
model.summary()

# callbacks
#check_point = ModelCheckpoint('keras_mnist_model.h5')
"""
cp_path = "test\\"
makedir(cp_path)
cp_path = 'test\\{epoch:02d}-{val_loss:.2f}.hdf5'
"""
check_point = ModelCheckpoint(filepath='keras_mnist_model_AlexNet.h5', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1).numpy()

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
print(round(model.optimizer.lr.numpy(), 5))

log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

csv_logger = CSVLogger('log.csv', append=True, separator=';')

# Fit the model
model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=200, callbacks=[check_point, early_stopping, lr_scheduler, csv_logger, tensor_board], verbose=1)

loaded_model = load_model('keras_mnist_model_AlexNet.h5')
loaded_model.summary()

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
scores_best = loaded_model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("CNN Error: %.2f%%" % (100-scores_best[1]*100))

# comparison between label y and predicted y
fig, axs = plt.subplots(1,5,figsize=(10,10))
for i in range(5):
    axs[i].imshow(X_test[i], cmap=plt.cm.binary)

plt.show()

print(y_test[:5])

import numpy as np
print(np.argmax(y_test[:80],axis=1))

#print(model.predict(x_test[0:5]))
print(np.argmax(model.predict(X_test[0:80]),axis=1))
