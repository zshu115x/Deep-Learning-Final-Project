from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, UpSampling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
from STL_loader import load_unlabeld_data, load_labeld_data
from keras.objectives import binary_crossentropy
from AutoEncoderLayers import DeConvAfterDownSampling
from theano.tensor.nnet.neighbours import images2neibs, neibs2images
from theano import typed_list, scan
import theano.tensor as T
import sys
import theano
import cPickle
sys.setrecursionlimit(10000)
theano.tensor.config.exception_verbosity = "high"


filter_size = 11
image_length = 96

# def nested_loop(y_true, y_pred):
#     outputs_info = T.zeros((y_true.shape[0], filter_size,
#                           filter_size , y_true.shape[3]))
#     def inner_loop(j, r, i):
#         y_true_sub = y_true[:, i:i + filter_size, j:j + filter_size, :]
#         y_pred_sub = y_pred[:, i * filter_size:(i + 1) * filter_size,
#                            j * filter_size:(j + 1) * filter_size, :]
#         r += K.square(y_pred_sub-y_true_sub)
#         return r
#
#     def outer_loop(i, r):
#         results, _ = scan(fn=inner_loop, sequences=[T.arange(image_length - filter_size + 1)],
#                           outputs_info=r, non_sequences=i)
#         results = results[-1]
#         return results
#
#     results, _ = scan(fn=outer_loop, sequences=[T.arange(image_length - filter_size + 1)],
#                       outputs_info=outputs_info)
#     final_results = results[-1]
#     return final_results


def inner_loop(j, r, i, y_true, y_pred):
    y_true_sub = y_true[:, i:i + filter_size, j:j + filter_size, :]
    y_pred_sub = y_pred[:, i * filter_size:(i + 1) * filter_size,
                 j * filter_size:(j + 1) * filter_size, :]
    r += K.square(y_pred_sub - y_true_sub)
    return r


def outer_loop(i, r, y_true, y_pred):
    results, _ = scan(fn=inner_loop, sequences=[T.arange(image_length - filter_size + 1)],
                      outputs_info=r, non_sequences=[i, y_true, y_pred])
    results = results[-1]
    return results


def Conv_AutoEncoder_mse(y_true, y_pred):
    outputs_info = T.zeros((y_true.shape[0], filter_size,
                            filter_size, y_true.shape[3]))

    results, _ = scan(fn=outer_loop, sequences=[T.arange(image_length - filter_size + 1)],
                  outputs_info=outputs_info, non_sequences=[y_true, y_pred])
    final_results = results[-1]
    # result = nested_loop(y_true, y_pred)
    return K.mean(final_results/((image_length - filter_size + 1)**2))

def Conv_AutoEncoder_mse_new(y_true, y_pred):
    y_true = y_true.dimshuffle(0, 3, 1, 2)
    y_true = images2neibs(y_true, neib_shape=(filter_size, filter_size), neib_step=(1, 1))

    return K.mean(K.square(y_pred[0][0]-y_true))
def train1():
    inputs = Input(shape=(96, 96, 1))
    encoder = Convolution2D(16, filter_size, filter_size, activation="sigmoid",
                            border_mode="valid", dim_ordering="tf")(inputs)

    # decoder = UpSampling2D(size=(filter_size, filter_size), dim_ordering="tf")(encoder)

    decoder = DeConvAfterDownSampling(1, filter_size, filter_size,
                                      activation="sigmoid",dim_ordering="tf")(encoder)

    model = Model(input=inputs, output=decoder)

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=Conv_AutoEncoder_mse_new)

    unlabeled_data = load_unlabeld_data(grayscale=True)  ### only load 2000 by default, if you wanna more,
    ### go to the function, and change the number
    train_data = unlabeled_data[:100]
    valid_data = unlabeled_data[100:200]
    # train_data = unlabeled_data[:90000]
    # # valid_data = unlabeled_data[80000:90000]
    # valid_data = unlabeled_data[90000:]
    model.fit(train_data, train_data, batch_size=5,
              nb_epoch=1, verbose=1, validation_data=(valid_data, valid_data))

    # save all weights
    model.save_weights("data/autoEncoderConv_weights.h5", overwrite=True)

    # save the architecture of the model
    json_string = model.to_json()
    open("data/autoEncoderConv_architecture.json", "w").write(json_string)

    # save encoder layer's weights
    encoder_weights = model.layers[1].get_weights()
    with open('data/encoder_weights.pickle', 'wb') as f:
        cPickle.dump(encoder_weights, f)
        f.close()

train1()

