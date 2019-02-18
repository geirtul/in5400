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

"""
Main routine capable of training a dense neural network, and also running inference.

This program builds an L-layer dense neural network. The number of nodes in each layer is set in
the configuration.

By default, every node has a ReLu activation, except the final layer, which has a softmax output.
We use a cross-entropy loss for the cost function, and we use a stochastic gradient descent
optimization routine to minimize the cost function.

Custom configuration for experimentation is possible.
"""

import os
import numpy as np

import import_data
import run

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def config():
    """Return a dict of configuration settings used in the program"""

    conf = {}

    # Determine what dataset to run on. 'mnist', 'cifar10' and 'svhn' are currently supported.
    conf['dataset'] = 'mnist'
    # Relevant datasets will be put in the location data_root_dir/dataset.
    conf['data_root_dir'] = "/tmp/data"

    # Number of input nodes. This is determined by the dataset in runtime.
    conf['input_dimension'] = None
    # Number of hidden layers, with the number of nodes in each layer.
    conf['hidden_dimensions'] = [128, 32]
    # Number of classes. This is determined by the dataset in runtime.
    conf['output_dimension'] = None
    # This will be determined in runtime when input_dimension and output_dimension is set.
    conf['layer_dimensions'] = None

    # Size of development partition of the training set
    conf['devel_size'] = 5000
    # What activation function to use in the nodes in the hidden layers.
    conf['activation_function'] = 'relu'
    # The number of steps to run before termination of training. One step is one forward->backward
    # pass of a mini-batch
    conf['max_steps'] = 2000
    # The batch size used in training.
    conf['batch_size'] = 128
    # The step size used by the optimization routine.
    conf['learning_rate'] = 1.0e-2

    # Whether or not to write certain things to stdout.
    conf['verbose'] = False
    # How often (in steps) to log the training progress. Prints to stdout if verbose = True.
    conf['train_progress'] = 10
    # How often (in steps) to evaluate the method on the development partition. Prints to stdout
    # if verbose = True.
    conf['devel_progress'] = 100

    return conf

def plot_progress(train_progress, devel_progress, out_filename=None):
    """Plot a chart of the training progress"""

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    ax1.plot(train_progress['steps'], train_progress['ccr'], 'b', label='Training set ccr')
    ax1.plot(devel_progress['steps'], devel_progress['ccr'], 'r', label='Development set ccr')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Correct classification rate')
    ax1.legend(loc='lower left', bbox_to_anchor=(0.6, 0.52), framealpha=1.0)

    ax2 = ax1.twinx()
    ax2.plot(train_progress['steps'], train_progress['cost'], 'g', label='Training set cost')
    ax2.set_ylabel('Cross entropy cost')
    gl2 = ax2.get_ygridlines()
    for gl in gl2:
        gl.set_linestyle(':')
        gl.set_color('k')

    ax2.legend(loc='lower left', bbox_to_anchor=(0.6, 0.45), framealpha=1.0)
    plt.title('Training progress')
    fig.tight_layout()

    if out_filename is not None:
        plt.savefig(out_filename)

    plt.show()

def get_data(conf):
    """Return data to be used in this session.

    Args:
        conf: Configuration dictionary
    Returns:
        X_train: numpy array of floats with shape [input_dimension, num train examples] in [0, 1].
        Y_train: numpy array of integers with shape [output_dimension, num train examples].
        X_devel: numpy array of floats with shape [input_dimension, num devel examples] in [0, 1].
        Y_devel: numpy array of integers with shape [output_dimension, num devel examples].
        X_test: numpy array of floats with shape [input_dimension, num test examples] in [0, 1].
        Y_test: numpy array of integers with shape [output_dimension, num test examples].
    """

    data_dir = os.path.join(conf['data_root_dir'], conf['dataset'])
    if conf['dataset'] == 'cifar10':
        conf['input_dimension'] = 32*32*3
        conf['output_dimension'] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_cifar10(
            data_dir, conf['devel_size'])
    elif conf['dataset'] == 'mnist':
        conf['input_dimension'] = 28*28*1
        conf['output_dimension'] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_mnist(
            data_dir, conf['devel_size'])
    elif conf['dataset'] == 'svhn':
        conf['input_dimension'] = 32*32*3
        conf['output_dimension'] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_svhn(
            data_dir, conf['devel_size'])

    conf['layer_dimensions'] = ([conf['input_dimension']] +
                                conf['hidden_dimensions'] +
                                [conf['output_dimension']])

    if conf['verbose']:
        print("Train dataset:")
        print("  shape = {}, data type = {}, min val = {}, max val = {}".format(X_train.shape,
                                                                                X_train.dtype,
                                                                                np.min(X_train),
                                                                                np.max(X_train)))
        print("Development dataset:")
        print("  shape = {}, data type = {}, min val = {}, max val = {}".format(X_devel.shape,
                                                                                X_devel.dtype,
                                                                                np.min(X_devel),
                                                                                np.max(X_devel)))
        print("Test dataset:")
        print("  shape = {}, data type = {}, min val = {}, max val = {}".format(X_test.shape,
                                                                                X_test.dtype,
                                                                                np.min(X_test),
                                                                                np.max(X_test)))

    return X_train, Y_train, X_devel, Y_devel, X_test, Y_test


def main():
    """Run the program according to specified configurations."""

    conf = config()

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))

if __name__ == "__main__":
    main()
