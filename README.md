# Noisy ECG AF Detection Reference Model v1

## Content

The provided script `main_reference_model.py` contains:

- a convolutional neural network architecture for distinguishing atrial fibrillation from sinus rhythm in noisy ECG data,
- a main function that (optionally) trains, saves and evaluates a (trained) model with the provided data set,
- utility methods to load a pickled data set and to make an evaluation report.

A pretrained model called `pretrained-reference-model.h5` is also available. It has been trained on a real ECG data set with classes atrial fibrillation and sinus rhythm, affected by various types of noise that can be found in ECG data. A training and evaluation log can be found in `pretrained-reference-model_30sec.log`.

## Usage

First, install the necessary libraries:

    $ pip install tensorflow scikit-learn matplotlib

The script expects a pickle (`*.pkl`) file as its command line argument.
You can execute it like this:

    $ python3 main_reference_model.py my_dataset.pkl

If you do not want to train your model and just evaluate it on a given data set, please set the `train` argument of the `main` function to `False`. The model called `saved-reference-model.h5` will be loaded each time for evaluation.

## Input data

The network is optimized to work with ECG data sampled at 128 Hz. Ideally, your input data is sampled at the same rate. Please adjust the `sample_rate` variable in the script otherwise.

The time frame to reliably detect atrial fibrillation is usually regarded to be 30 s. However, the time frame analysed by the neural network can be adjusted via the `time_frame` variable.

The pickle file must contain a `list` of two `numpy` arrays, consisting of the input data `x` (i.e. samples) and the corresponding expected output data (i.e. labels) `y` with `x.shape == (num_samples, sample_length, 1)` and `y.shape == (num_samples,)`.
