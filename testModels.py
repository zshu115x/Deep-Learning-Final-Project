from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D
from keras.optimizers import SGD
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from STL_loader import load_unlabeld_data, load_labeld_data, grayScaler

filter_size = 11

data = load_unlabeld_data(grayscale=False)

def construct_encoder():
    f = open("data/encoder_weights.pickle", "rb")
    encoder_weights = cPickle.load(f)

    # print np.shape(encoder_weights[1])

    model = Sequential()
    model.add(Convolution2D(16, filter_size, filter_size, input_shape=(96, 96, 1),
                            activation="sigmoid", border_mode="valid", dim_ordering="tf",
                            weights=encoder_weights, trainable=False))

    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd, loss="msle")

    orig_img = data[0]
    plt.subplot(542)
    plt.imshow(orig_img)
    gray_img = grayScaler(orig_img)
    plt.subplot(543)
    plt.imshow(np.rollaxis(gray_img, -1, 0)[0], cmap=plt.cm.gray)

    test_one = gray_img.reshape((1, gray_img.shape[0], gray_img.shape[1], gray_img.shape[2]))

    pred_one = model.predict(test_one, batch_size=1, verbose=1)

    feature_maps = np.rollaxis(pred_one, -1, 0)

    for i in xrange(len(feature_maps)):
        plt.subplot(5, 4, 5+i)
        plt.imshow(feature_maps[i][0], cmap=plt.cm.gray)
    plt.show()

def construct_autoEncoder():
    """
    Forget about it, I can't show it
    :return:
    """
    # model = Sequential()
    model = model_from_json(open("data/autoEncoderConv_architecture.json").read)
    model.load_weights("data/autoEncoderConv_weights.h5")

    test_data = grayScaler(data[:8])

    pred_data = model.predict(test_data, batch_size=8, verbose=1)

    # test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2]))
    # for i in xrange(len(test_data)):
    #     plt.subplot(3, 8, i+1)
    #     plt.imshow(data[i])
    #     plt.subplot(3, 8, 8+i+1)
    #     plt.imshow(test_data[i])
    #     plt.imshow()

