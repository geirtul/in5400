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

"""Module for training and evaluation"""

import time

import numpy as np

import model

def get_batch_indices(indices, start_index, end_index):
    """Return the indices of the examples that are to form a batch.

    This is done so that if end_index > len(example_indices), we will include the remainding
    indices, in addition to the first indices in the example_indices list.

    Args:
        indices: 1D numpy array of integers
        start_index: integer > 0 and smaller than len(example_indices)
        end_index: integer > start_index
    Returns:
        1D numpy array of integers
    """
    n = len(indices)
    return np.hstack((indices[start_index:min(n, end_index)], indices[0:max(end_index-n, 0)]))

def train(conf, X_train, Y_train, X_devel, Y_devel):
    """Run training

    Args:
        conf: Configuration dictionary
        X_train: numpy array of floats with shape [input dimension, number of train examples]
        Y_train: numpy array of integers with shape [output dimension, number of train examples]
        X_devel: numpy array of floats with shape [input dimension, number of devel examples]
        Y_devel: numpy array of integers with shape [output dimension, number of devel examples]
    Returns:
        params: Dictionary with trained parameters
        train_progress: Dictionary with progress data, to be used in visualization.
        devel_progress: Dictionary with progress data, to be used in visualization.
    """
    print("Run training")

    # Preparation
    num_examples_in_epoch = X_train.shape[1]
    example_indices = np.arange(0, num_examples_in_epoch)
    np.random.shuffle(example_indices)

    # Initialisation
    params = model.initialization(conf)

    # For displaying training progress
    train_steps = []
    train_ccr = []
    train_cost = []
    devel_steps = []
    devel_ccr = []

    # Start training
    step = 0
    epoch = 0
    num_correct_since_last_check = 0
    batch_start_index = 0
    batch_end_index = conf['batch_size']
    print("Number of training examples in one epoch: ", num_examples_in_epoch)
    print("Start training iteration")
    while True:
        start_time = time.time()
        batch_indices = get_batch_indices(example_indices, batch_start_index, batch_end_index)
        X_batch = X_train[:, batch_indices]
        Y_batch = model.one_hot(Y_train[batch_indices], conf['output_dimension'])

        Y_proposal, features = model.forward(conf, X_batch, params, is_training=True)
        cost_value, num_correct = model.cross_entropy_cost(Y_proposal, Y_batch)
        grad_params = model.backward(conf, Y_proposal, Y_batch, params, features)
        params = model.gradient_descent_update(conf, params, grad_params)

        num_correct_since_last_check += num_correct

        batch_start_index += conf['batch_size']
        batch_end_index += conf['batch_size']
        if batch_start_index >= num_examples_in_epoch:
            epoch += 1
            np.random.shuffle(example_indices)
            batch_start_index = 0
            batch_end_index = conf['batch_size']

        step += 1

        if np.isnan(cost_value):
            print("ERROR: nan encountered")
            break


        if step % conf['train_progress'] == 0:
            elapsed_time = time.time() - start_time
            sec_per_batch = elapsed_time / conf['train_progress']
            examples_per_sec = conf['batch_size']*conf['train_progress'] / elapsed_time
            ccr = num_correct / conf['batch_size']
            running_ccr = (num_correct_since_last_check /
                           conf['train_progress'] / conf['batch_size'])
            num_correct_since_last_check = 0
            train_steps.append(step)
            train_ccr.append(running_ccr)
            train_cost.append(cost_value)
            if conf['verbose']:
                print("S: {0:>7}, E: {1:>4}, cost: {2:>7.4f}, CCR: {3:>7.4f} ({4:>6.4f}),  "
                      "ex/sec: {5:>7.3e}, sec/batch: {6:>7.3e}".format(step, epoch, cost_value,
                                                                       ccr, running_ccr,
                                                                       examples_per_sec,
                                                                       sec_per_batch))

        if step % conf['devel_progress'] == 0:
            num_correct, num_evaluated = evaluate(conf, params, X_devel, Y_devel)
            devel_steps.append(step)
            devel_ccr.append(num_correct / num_evaluated)
            if conf['verbose']:
                print("S: {0:>7}, Test on development set. CCR: {1:>5} / {2:>5} = {3:>6.4f}".format(
                    step, num_correct, num_evaluated, num_correct/num_evaluated))

        if step >= conf['max_steps']:
            print("Terminating training after {} steps".format(step))
            break

    train_progress = {'steps': train_steps, 'ccr': train_ccr, 'cost': train_cost}
    devel_progress = {'steps': devel_steps, 'ccr': devel_ccr}

    return params, train_progress, devel_progress


def evaluate(conf, params, X_data, Y_data):
    """Evaluate a trained model on X_data.

    Args:
        conf: Configuration dictionary
        params: Dictionary with parameters
        X_data: numpy array of floats with shape [input dimension, number of examples]
        Y_data: numpy array of integers with shape [output dimension, number of examples]
    Returns:
        num_correct_total: Integer
        num_examples_evaluated: Integer
    """

    num_examples = X_data.shape[1]
    num_examples_evaluated = 0
    num_correct_total = 0
    start_ind = 0
    end_ind = conf['batch_size']
    while True:
        X_batch = X_data[:, start_ind: end_ind]
        Y_batch = model.one_hot(Y_data[start_ind: end_ind], conf['output_dimension'])
        Y_proposal, _ = model.forward(conf, X_batch, params, is_training=False)
        _, num_correct = model.cross_entropy_cost(Y_proposal, Y_batch)
        num_correct_total += num_correct

        num_examples_evaluated += end_ind - start_ind

        start_ind += conf['batch_size']
        end_ind += conf['batch_size']

        if end_ind >= num_examples:
            end_ind = num_examples

        if start_ind >= num_examples:
            break

    return num_correct_total, num_examples_evaluated
