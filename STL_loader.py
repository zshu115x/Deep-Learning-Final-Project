"""
load STL-10 binary data set
"""
import gzip
import tarfile
import numpy as np

# from theano import function as T_function
# import theano.tensor as T

import stl10_input

train_X_path = "data/stl10_binary/train_X.bin"
train_y_path = "data/stl10_binary/train_y.bin"
unlabeled_X_path = "data/stl10_binary/unlabeled_X.bin"
test_X_path = "data/stl10_binary/test_X.bin"
test_y_path = "data/stl10_binary/test_y.bin"

def unzip_data():
    """
    unzip the .gz file to "data/" directory, then we will get the paths as above
    :return:
    """
    tarfile.open("data/stl10_binary.tar.gz", 'r:gz').extractall("data/")

def load_labled_data(grayscale = False):
    """
    The value of labels should start from 0
    Do we need to normalize the data to be between 0-1? no need currently, otherwise double value needs more memory
    Grayscale the images
    :return:
    """
    train_inputs = stl10_input.read_all_images(train_X_path)
    train_labels = stl10_input.read_labels(train_y_path)
    test_inputs = stl10_input.read_all_images(test_X_path)
    test_labels = stl10_input.read_labels(test_y_path)
    if grayscale:
        return grayScaler(train_inputs)/255.0, train_labels - 1, grayScaler(test_inputs)/255.0, test_labels - 1
    else:
        return train_inputs/255.0, train_labels - 1, test_inputs/255.0, test_labels - 1

def load_unlabeld_data():
    data = stl10_input.read_all_images(unlabeled_X_path)
    return data

# images = stl10_input.read_labels(train_y_path)
# print images.shape

# new_images = np.dot(images, np.array([0.299, 0.587, 0.114]))
# print new_images.shape

def grayScaler(data):
    """
    the shape of data should be m*N*N*3, where m is
    the number of observations, N is the size of each image
    GrayScale = 0.299R + 0.587G + 0.114B, which is the formula
    to transform RGB to gray scale
    :param data:
    :return:
    """
    return np.dot(data, np.array([0.299, 0.587, 0.114]))

# train_inputs, train_labels, test_inputs, test_labels = load_labled_data(grayscale=True)
#
# stl10_input.plot_image(train_inputs[0])
#
# stl10_input.plot_image(np.transpose(train_inputs[0])[0])

# [np.transpose(x) for x in train_inputs]

# print train_inputs.shape


