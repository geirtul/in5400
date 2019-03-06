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

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    layer_dimensions = conf["layer_dimensions"]
    params = {}

    for i in range(len(layer_dimensions) - 1):
        layer_shape = (layer_dimensions[i], layer_dimensions[i + 1])
        layer_var = 2/layer_dimensions[i]

        weights = np.random.normal(0, np.sqrt(layer_var), layer_shape)
        bias = np.zeros((layer_shape[1], 1))

        params["W_{}".format(i+1)] = weights
        params["b_{}".format(i+1)] = bias

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        """
        in place modification seems to be the fastest by one OOM compared
        to vanilla np.maximum, x * (x > 0 )
        ref:
        https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy

        but is impractical without a rework of the intended architecture
        """
        a = Z * (Z > 0)
        return a
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z, axis=0):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 b)

    # again, in-place modification for speed
    x = np.exp(Z)
    normalization = np.sum(x, axis=axis, keepdims=True)

    return x/normalization


def forward(conf, X_batch, params, is_training, features=None):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c)

    features = {}
    num_iter = len(conf["layer_dimensions"]) - 2
    x = X_batch

    features["A_0"] = X_batch

    for i in range(num_iter):
        b = params["b_{}".format(i + 1)]
        w = params["W_{}".format(i + 1)]

        z_tmp = np.einsum("ib, io -> ob", x, w) + b
        a_tmp = activation(z_tmp, conf["activation_function"])
        features["Z_{}".format(i + 1)] = z_tmp
        features["A_{}".format(i + 1)] = a_tmp

        x = a_tmp

    b = params["b_{}".format(num_iter + 1)]
    w = params["W_{}".format(num_iter + 1)]

    z_tmp = np.einsum("ib, io -> ob", x, w) + b
    features["Z_{}".format(num_iter + 1)] = z_tmp

    Y_proposed = softmax(z_tmp)
    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference, treshold=0.5):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [m, n_y].
        Y_reference: numpy array of floats with shape [m, n_y]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    log_y_prop = np.log(Y_proposed)
    num_samples = Y_proposed.shape[1]

    cost = -(1/num_samples) * \
        np.einsum("ij, ij -> ...", Y_reference, log_y_prop)

    tmp = Y_proposed == Y_proposed.max(axis=0, keepdims=True)
    tmp = tmp.astype(np.int8)

    pred_labels = np.nonzero(tmp.T)[1]

    correct_labels = np.nonzero(Y_reference.T)[1]
    num_correct = np.sum(pred_labels == correct_labels)

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        return np.heaviside(Z, 1)
    else:
        print("Error: Unimplemented derivative of activation function: {}",
              activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    n_layers = len(conf["layer_dimensions"]) - 1
    batch_size = Y_proposed.shape[1]
    activation_function = conf["activation_function"]

    grad_params = {}

    # aM = features["A_{}".format(n_layers)] # aM is Y_proposed
    a_prev = features["A_{}".format(n_layers - 1)]

    dEdY = Y_proposed - Y_reference
    dEdX_cur = dEdY  # activation_derivative(zM, activation_function)
    dXdW_cur = a_prev

    dEdW_cur = np.einsum("ki, ji -> jk", dEdX_cur, dXdW_cur)
    dEdB_cur = np.expand_dims(np.einsum("ij -> i", dEdX_cur), axis=1)

    grad_params["grad_W_{}".format(n_layers)] = dEdW_cur/batch_size
    grad_params["grad_b_{}".format(n_layers)] = dEdB_cur/batch_size

    layer_iter = [i for i in range(n_layers-1)]
    layer_iter.reverse()

    for la in layer_iter:
        la += 1

        dXlpdAl = params["W_{}".format(la+1)]
        dAldXl = activation_derivative(
            features["Z_{}".format(la)], activation_function)
        dXldWl = features["A_{}".format(la-1)]

        #print(dEdX_cur.shape, dXlpdAl.shape)
        dEdAl = np.einsum("ki, jk -> ji", dEdX_cur, dXlpdAl)
        dEdXl = dEdAl * dAldXl
        #print(dXldWl.shape, dEdXl.shape)
        dEdWl = np.einsum("ji, hi -> hj", dEdXl, dXldWl)
        dEdBl = np.expand_dims(np.einsum("ij -> i", dEdXl), axis=1)

        #print(dEdWl.shape, params["W_"+str(la)].shape)
        grad_params["grad_W_{}".format(la)] = dEdWl/batch_size
        grad_params["grad_b_{}".format(la)] = dEdBl/batch_size

        dEdX_cur = dEdXl

    return grad_params


def gradient_descent_update(conf, params, grad_params, prev_grad_params=None):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    eta = conf["learning_rate"]
    updated_params = {}

    for key, value in params.items():
        dEdValue = grad_params["grad_"+key]
        updated_params[key] = value - eta*dEdValue

        if not prev_grad_params is None:
            updated_params[key] += - eta * 1.2 * prev_grad_params["grad_"+key]

    return updated_params
