#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Import data"""

import os
import tarfile
import gzip
import pickle
import scipy.io as scio

import requests
import numpy as np


def maybe_download(url, dest_dir):
    """Checks if file exists and tries to download it if not"""
    if not os.path.exists(dest_dir):
        print("Creating directory: {}".format(dest_dir))
        os.makedirs(dest_dir)
    filename = url.split("/")[-1]
    dest_filepath = os.path.join(dest_dir, filename)

    if not os.path.exists(dest_filepath):
        print("Downloading", url)
        response = requests.get(url)
        with open(dest_filepath, 'wb') as fil:
            fil.write(response.content)
    return dest_filepath


def load_mnist(data_dir="data/mnist", devel_size=10000):
    """
    Loads training, validation, and test partitions of the mnist dataset
    (http://yann.lecun.com/exdb/mnist/). If the data is not already contained in data_dir, it will
    try to download it.

    This dataset contains 60000 training examples, and 10000 test examples of handwritten digits
    in {0, ..., 9} and corresponding labels. Each handwritten image has an "original" dimension of
    28x28x1, and is stored row-wise as a string of 784x1 bytes. Pixel values are in range 0 to 255
    (inclusive).

    Args:
        data_dir: String. Relative or absolute path of the dataset.
        devel_size: Integer. Size of the development (validation) dataset partition.

    Returns:
        X_train: float64 numpy array with shape [784, 60000-devel_size] with values in [0, 1].
        Y_train: uint8 numpy array with shape [60000-devel_size]. Labels.
        X_devel: float64 numpy array with shape [784, devel_size] with values in [0, 1].
        Y_devel: uint8 numpy array with shape [devel_size]. Labels.
        X_test: float64 numpy array with shape [784, 10000] with values in [0, 1].
        Y_test: uint8 numpy array with shape [10000]. Labels.
    """

    def _load_data(filename, data_dir, header_size):
        """Load mnist images or labels. This tries to download the data if it is not found.

        Args:
            filename: Filename of the dataset. Is appended to the root url and used to download the
                      data if it is not already downloaded.
            data_dir: String. Destination directory.
            header_size: uint8. Size of the header in bytes, which is 8 for labels and 16 for
                         images. See the mnist webpage for more info.
        Returns:
            data: uint8 numpy array
        """
        url = "http://yann.lecun.com/exdb/mnist/" + filename
        data_filepath = maybe_download(url, data_dir)
        with gzip.open(data_filepath, 'rb') as fil:
            data = np.frombuffer(fil.read(), np.uint8, offset=header_size)
        return np.asarray(data, dtype=np.uint8)

    print("Loading MNIST data from ", data_dir)
    X_train = _load_data('train-images-idx3-ubyte.gz', data_dir, 16).reshape((-1, 784)).T
    Y_train = _load_data('train-labels-idx1-ubyte.gz', data_dir, 8)
    X_test = _load_data('t10k-images-idx3-ubyte.gz', data_dir, 16).reshape((-1, 784)).T
    Y_test = _load_data('t10k-labels-idx1-ubyte.gz', data_dir, 8)

    # Scale data to [0.0, 1.0]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Partition training into training and development set
    X_train, X_devel = X_train[:, :-devel_size], X_train[:, -devel_size:]
    Y_train, Y_devel = Y_train[:-devel_size], Y_train[-devel_size:]

    return X_train, Y_train, X_devel, Y_devel, X_test, Y_test


def load_cifar10(data_dir="data/cifar10", devel_size=10000):
    """
    Loads training, validation, and test partitions of the cifar10 dataset
    (https://www.cs.toronto.edu/~kriz/cifar.html). If the data is not already contained in
    data_dir, it will try to download it.

    Info from the webpage:
        The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
        per class. There are 50000 training images and 10000 test images.

        The dataset is divided into five training batches and one test batch, each with 10000
        images. The test batch contains exactly 1000 randomly-selected images from each class. The
        training batches contain the remaining images in random order, but some training batches
        may contain more images from one class than another. Between them, the training batches
        contain exactly 5000 images from each class.

    The cifar10 dataset contains the following classes

    Label   Description
    -------------------
    0       airplane
    1       automobile
    2       bird
    3       cat
    4       deer
    5       dog
    6       frog
    7       horse
    8       ship
    9       truck

    Args:
        data_dir: String. Relative or absolute path of the dataset.
        devel_size: Integer. Size of the development (validation) dataset partition.

    Returns:
        X_train: float64 numpy array with shape [3072, 60000-devel_size] with values in [0, 1].
        Y_train: uint8 numpy array with shape [60000-devel_size]. Labels.
        X_devel: float64 numpy array with shape [3072, devel_size] with values in [0, 1].
        Y_devel: uint8 numpy array with shape [devel_size]. Labels.
        X_test: float64 numpy array with shape [3072, 10000] with values in [0, 1].
        Y_test: uint8 numpy array with shape [10000]. Labels.
    """

    print("Loading cifar10 data from", data_dir)

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filepath = maybe_download(url, data_dir)
    if tarfile.is_tarfile(filepath):
        print('Extracting tar file:', filepath)
        tarfile.open(filepath, 'r').extractall(data_dir)
    else:
        print("Extraction of compression format is not implemented")

    #Unpickle file and fill in data
    X_train = None
    Y_train = []
    for ind in range(1, 6):
        filepath = os.path.join(data_dir, "cifar-10-batches-py/data_batch_{}".format(ind))
        with open(filepath, 'rb') as fil:
            data_dict = pickle.load(fil, encoding='latin-1')
        if ind == 1:
            X_train = data_dict['data']
        else:
            X_train = np.vstack((X_train, data_dict['data']))
        Y_train.extend(data_dict['labels'])
    Y_train = np.array(Y_train)

    with open(os.path.join(data_dir, "cifar-10-batches-py/test_batch"), 'rb') as fil:
        test_data_dict = pickle.load(fil, encoding='latin-1')
    X_test = test_data_dict['data']
    Y_test = np.array(test_data_dict['labels'])

    # Reshape data
    X_test = X_test.reshape((-1, 32*32*3)).T
    X_train = X_train.reshape((-1, 32*32*3)).T

    # Scale data to [0.0, 1.0]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Partition training into training and development set
    X_train, X_devel = X_train[:, :-devel_size], X_train[:, -devel_size:]
    Y_train, Y_devel = Y_train[:-devel_size], Y_train[-devel_size:]

    return X_train, Y_train, X_devel, Y_devel, X_test, Y_test

def load_svhn(data_dir, devel_size=10000):
    """Load the Street View House Numbers (SVHN) dataset

    http://ufldl.stanford.edu/housenumbers/

    Info from the website
        - 10 classes, 1 for each digit. Digit '1' has label 1, '9' has label 9 and '0' has label 10.
        - 73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat
          less difficult samples, to use as extra training data

    Each .mat file have an X and a y field:

        X: numpy array of uint8 with shape [32, 32, 3, num_images]
        y: numpy array of uint8 with shape [1, num_images]

    Args:
        data_dir: String. Relative or absolute path of the dataset.
        devel_size: Integer. Size of the development (validation) dataset partition.

    Returns:
        X_train: float64 numpy array with shape [3072, 73257-devel_size] with values in [0, 1].
        Y_train: uint8 numpy array with shape [73257-devel_size]. Labels.
        X_devel: float64 numpy array with shape [3072, devel_size] with values in [0, 1].
        Y_devel: uint8 numpy array with shape [devel_size]. Labels.
        X_test: float64 numpy array with shape [3072, 26032] with values in [0, 1].
        Y_test: uint8 numpy array with shape [26032]. Labels.
    """

    url = "http://ufldl.stanford.edu/housenumbers/"
    print("Loading svhn data from {}".format(data_dir))

    _ = maybe_download(url+"train_32x32.mat", data_dir)
    train_data = scio.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    rows, cols, channels, num_train = train_data['X'].shape
    X_train = train_data['X'].reshape((rows*cols*channels, num_train))
    Y_train = train_data['y'].squeeze()

    _ = maybe_download(url+"test_32x32.mat", data_dir)
    test_data = scio.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    rows, cols, channels, num_test = test_data['X'].shape
    X_test = test_data['X'].reshape((rows*cols*channels, num_test))
    Y_test = test_data['y'].squeeze()

    # Number 0 was initially labeled as 10
    Y_train[Y_train == 10] = 0
    Y_test[Y_test == 10] = 0

    # Scale data to [0.0, 1.0]. Other data standardisation can be done here.
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Partition training into training and development set
    X_train, X_devel = X_train[:, :-devel_size], X_train[:, -devel_size:]
    Y_train, Y_devel = Y_train[:-devel_size], Y_train[-devel_size:]

    return X_train, Y_train, X_devel, Y_devel, X_test, Y_test
