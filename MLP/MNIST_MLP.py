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
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
(X_train_origianl, y_train_original) = (X_train, y_train)
(X_test_original, y_test_original) = (X_test, y_test)
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
#(60000, 784) (60000,) (10000, 784) (10000,)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
y_train.shape

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

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
check_point = ModelCheckpoint(filepath='keras_mnist_model_MLP.h5', monitor='val_loss', save_best_only=True, verbose=1)
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
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=200, callbacks=[check_point, early_stopping, lr_scheduler, csv_logger, tensor_board], verbose=1)

round(model.optimizer.lr.numpy(), 5)

loaded_model = load_model('keras_mnist_model_MLP.h5')
loaded_model.summary()

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
scores_best = loaded_model.evaluate(X_test, y_test, verbose=0)
print("MLP Error: %.2f%%" % (100-scores[1]*100))
print("MLP Error: %.2f%%" % (100-scores_best[1]*100))
pred = model.predict(X_test)

# comparison between label y and predicted y
fig, axs = plt.subplots(1,5,figsize=(10,10))
for i in range(5):
    axs[i].imshow(X_test_original[i], cmap=plt.cm.binary)

plt.show()

print(y_test[:5])

import numpy as np
print(np.argmax(y_test[:80],axis=1))

#print(model.predict(x_test[0:5]))
print(np.argmax(pred[:80],axis=1))
