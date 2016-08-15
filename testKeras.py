
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn import cross_validation, metrics
from keras import backend as K

import cPickle

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

    model.add(Convolution2D(32, 11, 11, input_shape=(96, 96, n_channels),
                            activation="relu", border_mode="valid", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    # model.add(Dropout(0.25))


    model.add(Convolution2D(64, 10, 10, activation="relu", border_mode="valid", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(128, 8, 8, activation="relu", border_mode="valid", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(256, 2, 2, activation="relu", border_mode="valid", dim_ordering="tf"))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    model.add(Dropout(0.25))


    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation="softmax"))

    sgd = SGD(lr=0.25, momentum=0.9, nesterov=True)
    model.compile(loss="msle", optimizer=sgd, metrics=["accuracy"])
    return model

def data_loader(grayscale=False):
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


def run():
    model = creat_model()

    train_inputs, train_labels, test_inputs, test_labels = data_loader()

    valid_inputs = train_inputs[4000:]
    valid_labels = train_labels[4000:]
    train_inputs = train_inputs[:4000]
    train_labels = train_labels[:4000]

    hist = model.fit(train_inputs, train_labels, batch_size=50, nb_epoch=100,
                     verbose=1, validation_data=(valid_inputs, valid_labels))

    log = hist.history.values()

    with open('data/log/log_new_model_classification.pickle', 'w') as outfile:
        cPickle.dump(log, outfile)

    # test_pred = model.predict(test_inputs, batch_size=50)
    # print udf_matrix(udf_softmax(test_labels), udf_softmax(test_pred))

run()