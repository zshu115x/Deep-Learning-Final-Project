from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from STL_loader import grayScaler, load_unlabeld_data

import matplotlib.pyplot as plt
import numpy as np

import cPickle
def model1():
    f = open("data/encoder_weights_layer1_sigmoid.pickle", "rb")
    encoder_weights = cPickle.load(f)

    f.close()
    model = Sequential()

    model.add(Convolution2D(16, 11, 11, border_mode="valid", input_shape=(96, 96, 1),
                            activation="sigmoid", dim_ordering="tf", weights=encoder_weights, trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="msle", optimizer=sgd)
    return model

def model2():
    f = open("data/encoder_weights_layer1_sigmoid.pickle", "rb")
    encoder_weights = cPickle.load(f)

    f = open("data/encoder_weights_layer2_sigmoid.pickle", "rb")
    encoder_weights_layer2 = cPickle.load(f)

    f.close()

    model = Sequential()

    model.add(Convolution2D(16, 11, 11, border_mode="valid", input_shape=(96, 96, 1),
                            activation="sigmoid", dim_ordering="tf", weights=encoder_weights, trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(32, 10, 10, activation="sigmoid", dim_ordering="tf",
                            weights=encoder_weights_layer2, trainable=False))


    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss="msle", optimizer=sgd)
    return model

def model3():
    f = open("data/encoder_weights_layer1_sigmoid.pickle", "rb")
    encoder_weights = cPickle.load(f)

    f = open("data/encoder_weights_layer2_sigmoid.pickle", "rb")
    encoder_weights_layer2 = cPickle.load(f)

    f = open("data/encoder_weights_layer3_sigmoid.pickle", "rb")
    encoder_weights_layer3 = cPickle.load(f)

    f.close()

    model = Sequential()

    model.add(Convolution2D(16, 11, 11, border_mode="valid", input_shape=(96, 96, 1),
                            activation="sigmoid", dim_ordering="tf", weights=encoder_weights, trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(32, 10, 10, activation="sigmoid", dim_ordering="tf",
                            weights=encoder_weights_layer2, trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(64, 5, 5, activation="sigmoid", dim_ordering="tf",
                            weights=encoder_weights_layer3, trainable=False))
    model.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="tf"))

    sgd = SGD(lr=0.1, momentum=0.8, nesterov=True)
    model.compile(loss="msle", optimizer=sgd)
    return model

m1 = model1()
m2 = model2()
m3 = model3()

data = load_unlabeld_data(grayscale=False)

orig_img = data[3]
plt.subplot(8, 16, 8)
plt.imshow(orig_img, cmap=plt.cm.gray)
orig_shape = orig_img.shape

test = orig_img.reshape((1, orig_shape[0], orig_shape[1], orig_shape[2]))
test = grayScaler(test)

print test.shape
# 16 maps
pred1 = m1.predict(test, batch_size=1, verbose=1)
maps1 = np.rollaxis(pred1[0], -1, 0)
print maps1.shape
for i in xrange(maps1.shape[0]):
    plt.subplot(8, 16, 16+ i+1)
    plt.imshow(maps1[i], cmap=plt.cm.gray)

# 32 maps
pred2 = m2.predict(test, batch_size=1, verbose=1)
maps2 = np.rollaxis(pred2[0], -1, 0)
print maps2.shape
for i in xrange((maps2.shape[0])):
    plt.subplot(8, 16, 32+ i + 1)
    plt.imshow(maps2[i], cmap=plt.cm.gray)

# 64 maps
pred3 = m3.predict(test, batch_size=1, verbose=1)
maps3 = np.rollaxis(pred3[0], -1, 0)
print maps3.shape
for i in xrange((maps3.shape[0])):
    plt.subplot(8, 16, 64 + i + 1)
    plt.imshow(maps3[i], cmap=plt.cm.gray)

plt.show()