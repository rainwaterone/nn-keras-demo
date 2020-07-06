# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:56:55 2020

@author: rainwater-e
ref: https://keras.io/examples/vision/mnist_convnet/
"""
import time
start_time = time.time()
from AdvancedAnalytics.NeuralNetwork import nn_keras, nn_classifier
from AdvancedAnalytics.Regression    import logreg

import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.linear_model   import LogisticRegression
from sklearn.neural_network import MLPClassifier 

## import  and set up CUDA stuff
# from tensorflow.python.client import device_lib
# config = tf.ConfigProto(device_count={'GPU':1, 'CPU':12})
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)




def header(headerstring):
    lead = 38-(int(len(headerstring)/2))
    tail = 76-lead-len(headerstring)
    print('\n' + ('*'*78))
    print(('*'*lead) + ' ' + headerstring + ' ' + ('*'*tail))
    print(('*'*78))
    return

header('PREPROCESSING...')

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

num_pixels = x_train.shape[1] * x_train.shape[2]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


batch_size = 128
epochs = 44



# header('Running Keras Sequential with 2DCNN...')
# tf.keras.backend.clear_session()
# with tf.device('/GPU:0'):
#     model = tf.keras.Sequential()
#     # model.add(tf.keras.Input(shape=input_shape))
#     model.add(layers.Conv2D(32, kernel_size=(3, 3), 
#                             input_shape = input_shape, activation="relu"))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
#     model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(num_classes, activation="softmax"))
    
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    
#     history = model.fit(x_train, y_train, validation_split=0.1,
#               epochs=epochs, batch_size=200, verbose=1, shuffle=1)

# model.summary()

# nn_keras.accuracy_plot(history.history)
# # nn_keras.display_metrics(model, x_train, y_train)
# # nn_keras.display_metrics(model, x_test, y_test)
# score = model.evaluate(x_test, y_test, verbose=0)

# print("Keras CNN Test loss:", score[0])
# print("Keras CNN Test accuracy:", score[1])

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['loss', 'val_loss'])
# plt.title('CNN - Loss')
# plt.xlabel('epoch')

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.legend(['acc', 'val_acc'])
# plt.title('CNN - Accuracy')
# plt.xlabel('epoch')

# Attempt to run a model with conventional dense layers
header('Running Keras Sequential with Dense Layers...')
tf.keras.backend.clear_session()

hlaf = 'relu' # Hidden Layer Activation Function

npl = 1024

model = keras.Sequential()
model.add(layers.Dense(npl, input_dim = num_pixels,
                activation = hlaf))
model.add(layers.Dense(npl, activation = hlaf))       # Adds another layer
model.add(layers.Dense(npl, activation = hlaf))       # And another
model.add(layers.Dense(npl, activation = hlaf))       # And another
model.add(layers.Dense(npl, activation = hlaf))       # Adds another layer
model.add(layers.Dense(npl, activation = hlaf))       # And another
model.add(layers.Dense(npl, activation = hlaf))       # And another
model.add(layers.Dense(num_classes, activation = 'softmax'))
model.compile(optimizers.Adam(lr=0.01),
            loss='categorical_crossentropy',
            metrics=['acc'])

print(model.summary())

#Must flatten the model before fitting Dense layers

print('\nInitial x_train shape: ', x_train.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print('Reshaped x_train shape: ', x_train.shape)
print('Reshaped x_test  shape: ', x_test.shape)

history = model.fit(x_train, y_train, validation_split=0.1,
         epochs=epochs, batch_size=200, verbose=1, shuffle=1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Keras Dense Layer Test loss:", score[0])
print("Keras Dense Layer Test accuracy:", score[1])
nn_keras.accuracy_plot(history.history)
nn_keras.display_metrics(model, x_train, y_train)
nn_keras.display_metrics(model, x_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('Dense Layer - Loss')
plt.xlabel('epoch')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Dense Layer - Accuracy')
plt.xlabel('epoch')

# header('Running scikit-learn logistic regression...')

# # Whoops, we gotta re-shape the y tensor
# y_train=np.argmax(y_train, axis=1, out=None)
# y_test=np.argmax(y_test, axis=1, out=None)

# lr  = LogisticRegression(C=np.inf, tol=1e-16, max_iter=250)
# lr  = lr.fit(x_train, y_train)
# print("\nSci-Kit Learn Logistic Regression Model with C=np.inf")
# logreg.display_metrics(lr, x_train, y_train)

# header('Running scikit-learn FNN...')

# fnn = MLPClassifier(hidden_layer_sizes=(3,2), tol=1e-16,\
#                     activation="tanh", max_iter=250, \
#                     solver='lbfgs', random_state=12345)
# fnn = fnn.fit(x_train, y_train)
# print("\nSci-Kit Learn Neural Network with Two Layers 3/2")
# nn_classifier.display_metrics(fnn, x_train, y_train)



exetimestring = 'Execution Time: ' + str(round(time.time()-start_time,2)) + \
 ' seconds'
print(exetimestring)

