from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from STL_loader import load_unlabeld_data, load_labeld_data, grayScaler
from testKeras import show_confusionMatrix, data_loader, udf_matrix, udf_softmax

filter_size = 11

data = load_unlabeld_data(grayscale=False)

def plot_feature_maps(data, model):

    orig_img = data
    plt.subplot(542)
    plt.imshow(orig_img)
    gray_img = grayScaler(orig_img)
    plt.subplot(543)
    plt.imshow(np.rollaxis(gray_img, -1, 0)[0], cmap=plt.cm.gray)

    test_one = gray_img.reshape((1, gray_img.shape[0], gray_img.shape[1], gray_img.shape[2]))

    pred_one = model.predict(test_one, batch_size=1, verbose=1)

    feature_maps = np.rollaxis(pred_one, -1, 0)

    for i in xrange(len(feature_maps)):
        plt.subplot(5, 4, 5 + i)
        plt.imshow(feature_maps[i][0], cmap=plt.cm.gray)
    plt.show()

def construct_encoder():
    f = open("data/model_weights/encoder_weights.pickle", "rb")
    encoder_weights = cPickle.load(f)

    # print np.shape(encoder_weights[1])

    model = Sequential()
    model.add(Convolution2D(16, filter_size, filter_size, input_shape=(96, 96, 1),
                            activation="sigmoid", border_mode="valid", dim_ordering="tf",
                            weights=encoder_weights, trainable=False))

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="msle")

    model.save_weights("data/encoder_model_weights.h5", overwrite=True)

    json_string = model.to_json()
    open("data/encoder_architecture.json", "w").write(json_string)

    plot_feature_maps(data[3], model)

def construct_autoEncoder():
    """
    Forget about it, I can't show it
    :return:
    """
    # model = Sequential()
    model = model_from_json(open("data/autoEncoderConv_architecture.json").read())
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

def construct_feature_maps():
    model = model_from_json(open("data/encoder_architecture.json").read())
    model.load_weights("data/encoder_model_weights.h5")

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss="mse")

    plot_feature_maps(data[3], model)

def load_loss_log():
    f = open("data/model_weights/log.pickle", "rb")
    log = cPickle.load(f)
    print log

def creat_classification_model():
    f = open("data/encoder_weights.pickle", "rb")
    encoder_weights = cPickle.load(f)

    model = Sequential()

    model.add(Convolution2D(16, 11, 11, border_mode="valid", input_shape=(96, 96, 1),
                            activation="sigmoid", dim_ordering="tf", weights=encoder_weights, trainable=False))
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

def run(train_inputs, train_labels, valid_inputs, valid_labels):
    model = creat_classification_model()

    hist = model.fit(train_inputs, train_labels, batch_size=50, nb_epoch=50,
                     verbose=1, validation_data=(valid_inputs, valid_labels))
    model.save_weights("data/classification_model_weights")
    json_string = model.to_json()
    open("data/classification_model_architecture.json", "w").write(json_string)

    log = hist.history.values()
    with open('data/log_classification.pickle', 'wb') as f:
        cPickle.dump(log, f)
        f.close()
    # test_pred = model.predict(test_inputs, batch_size=50)
    # print udf_matrix(udf_softmax(test_labels), udf_softmax(test_pred))

train_inputs, train_labels, test_inputs, test_labels = data_loader(grayscale=True)

run(train_inputs, train_labels, test_inputs, test_labels)
