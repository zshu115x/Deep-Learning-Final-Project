#!/usr/share/anaconda2/bin/python

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import cross_validation

from STL_loader import load_labled_data

import numpy as np

grayscale = False
n_channels = 3
if grayscale:
    n_channels = 1

def creat_model():
    model = Sequential()

    model.add(Convolution2D(16, 11, 11, border_mode="valid", input_shape=(n_channels, 96, 96)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 10, 10))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 5, 5))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="msle", optimizer=sgd, metrics = ["accuracy"])
    return model


train_inputs, train_labels, test_inputs, test_labels = load_labled_data(grayscale)


# convert class vactors to binary class matrices
if grayscale:
    train_inputs = train_inputs.reshape(len(train_inputs),n_channels,96,96)
else:
    train_inputs = np.array([[z.transpose() for z in x.transpose()] for x in train_inputs])

train_labels = np_utils.to_categorical(train_labels, 10)


# test_inputs = test_inputs.reshape(len(test_inputs), 1, 96, 96)
# test_labels = np_utils.to_categorical(test_labels, 10)

cv = cross_validation.KFold(len(train_labels), n_folds = 5, shuffle=True)


i = 1
for trainCV, validCV in cv:
    print "fold {}".format(i)
    cv_train_inputs = train_inputs[trainCV]
    cv_train_labels = train_labels[trainCV]
    cv_valid_inputs = train_inputs[validCV]
    cv_valid_labels = train_labels[validCV]
    model = creat_model()
    model.fit(cv_train_inputs, cv_train_labels,
          batch_size=50, nb_epoch=50, verbose=1, show_accuracy=True, validation_data=(cv_valid_inputs, cv_valid_labels))
    i += 1



# model.fit(train_inputs, train_labels,
#           batch_size=20, nb_epoch=20, verbose=1, show_accuracy=True, validation_data=(test_inputs, test_labels))
