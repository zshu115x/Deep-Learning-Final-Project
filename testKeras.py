
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import cross_validation, metrics
from keras import backend as K

from STL_loader import load_labeld_data

import numpy as np
from pandas import DataFrame
import json

grayscale = False
n_channels = 3
if grayscale:
    n_channels = 1

label_names=["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

def udf_softmax(output):
    return [np.argmax(x) for x in output]

def udf_matrix(y_true, y_pred):
    matrix = metrics.confusion_matrix(y_true, y_pred)
    return DataFrame(matrix, index=label_names, columns=label_names)

def show_confusionMatrix(model, inputs, labels):
    """
    show confusionMatrix given the model, inputs and the corresonding labels
    :param model:
    :param inputs:
    :param labels:
    :return:
    """
    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()], model.layers[-1].output)
    output = get_last_layer_output([inputs, 0])

    print udf_matrix(udf_softmax(labels), udf_softmax(output))

def creat_model():
    model = Sequential()

    model.add(Convolution2D(16, 11, 11, border_mode="valid", input_shape=(96, 96, n_channels),
                            activation="relu", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    model.add(Convolution2D(32, 10, 10, activation="relu", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    model.add(Convolution2D(64, 5, 5, activation="relu", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="tf"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(10, activation="softmax"))

    sgd = SGD(lr=0.1, momentum=0.8, nesterov=True)
    model.compile(loss="msle", optimizer=sgd, metrics=["accuracy"])
    return model

def data_loader():
    train_inputs, train_labels, test_inputs, test_labels = load_labeld_data(grayscale)

    # if grayscale:
    #     train_inputs = train_inputs.reshape(len(train_inputs),n_channels,96,96)
    #     test_inputs = test_inputs.reshape(len(test_inputs), n_channels, 96, 96)
    # else:
    #     """transform (96, 96, 3) to (3, 96, 96)"""
    #     train_inputs = np.rollaxis(train_inputs, axis = 3, start=1)
    #     test_inputs = np.rollaxis(test_inputs, axis = 3, start=1)
    # # convert class vactors to binary class matrices
    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    return train_inputs, train_labels, test_inputs, test_labels


def CV_run(train_inputs, train_labels, n_folds = 10):
    cv = cross_validation.KFold(len(train_labels), n_folds = n_folds, shuffle=True)
    hists_cv = []
    i = 1
    for trainCV, validCV in cv:
        print "fold {}".format(i)
        cv_train_inputs = train_inputs[trainCV]
        cv_train_labels = train_labels[trainCV]
        cv_valid_inputs = train_inputs[validCV]
        cv_valid_labels = train_labels[validCV]
        model = creat_model()
        hist = model.fit(cv_train_inputs, cv_train_labels,
              batch_size=50, nb_epoch=20, verbose=1, validation_data=(cv_valid_inputs, cv_valid_labels))

        hists_cv.append(hist.history.values())
        i += 1
    with open('data/log.txt', 'w') as outfile:
        json.dump(hists_cv, outfile)


def run(train_inputs, train_labels, test_inputs, test_labels):
    model = creat_model()

    hist = model.fit(train_inputs, train_labels, batch_size=50, nb_epoch=20,
                     verbose=1, validation_data=(test_inputs, test_labels))

    show_confusionMatrix(model, test_inputs, test_labels)

    # with open('data/log.txt', 'w') as outfile:
    #     json.dump(hist, outfile)

train_inputs, train_labels, test_inputs, test_labels =  data_loader()

run(train_inputs, train_labels, test_inputs, test_labels)