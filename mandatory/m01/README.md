# Mandatory assignment 1

IN5400/IN9400 - Machine Learning for Image Analysis
University of Oslo
Spring 2019

## Content

Everything you need for this exercise is contained in this folder. A brief description of the
content follows.

## Submission
Once you are done working, run the collectSubmission.sh script ( on Linux); this will produce a file called IN5400_assignment1.zip. On Windows, use e.g. 7-zip. Do not include the data directories in your zip file. 

Then upload the zip-file file to devilry (devilry.ifi.uio.no). You can make multiple submissions before the deadline.

Good luck!




### `uio_in5400_s2019_mandatory1_assignment.ipynb`

Everything related to the assignment. This should be self-contained, and all information is found
in this notebook. You can start the notebook from the command line with

```
$ jupyter notebook uio_in5400_s2019_mandatory1_assignment.ipynb
```

### Content of supplied code

The exercise contains this notebook and two folders with code: `dnn` and `cnn`:

```
 → tree
.
├── cnn
│   ├── conv_layers.py
│   └── __init__.py
├── dnn
│   ├── import_data.py
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── run.py
│   └── tests.py
├── figures
│   ├── backprop_conv.png
│   ├── cifar10_progress_default.png
│   ├── convolution_same.png
│   ├── convolution_same_x11.png
│   ├── convolution_same_x12.png
│   ├── convolution_same_x33.png
│   ├── mnist_progress_default.png
│   └── svhn_progress_default.png
├── README.md
└── uio_in5400_s2019_mandatory1_assignment.ipynb
```

### `dnn`

This folder contains the whole dense neural network program. All functions that you are to
implement in this exercise is found in `dnn/model.py`, but you are of course free to edit
everything you want.

When you have implemented everything, you should be able to test your classifier with

```
$ python dnn/main.py
```

#### `main.py`

Handles program flow, data input and configurations. You should be able to run this file as an
executable: `$ python dnn/main.py`.

You should not need to change anything here.


#### `import_data.py`

Handles import of the following three datasets

- MNIST
- CIFAR10
- SVHN

You should not need to change anything here.


#### `run.py`

Contains training and evaluation routines.

You should not need to change anything here

#### `model.py`

Implements all the important logic of the classifier.

Everything you need to implement will be located in this file.

#### `tests.py`

In this file, predefined arrays are defined. To be used when checking your implementations.

### `cnn`

In this folder, code related to the convolutional neural network implementations reside.
