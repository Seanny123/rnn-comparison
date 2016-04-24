import lstm_test
import pickle
import numpy as np

import matplotlib.pyplot as plt

import ipdb

with open("mnist.pkl", "rb") as f:
    train, _, test = pickle.load(f)

train_targets = np.zeros((train[1].shape[0], 1, 10), dtype=np.float32)
train_targets[np.arange(train[1].shape[0]), :, train[1]] = 1.0

test_targets = np.zeros((test[1].shape[0], 1, 10), dtype=np.float32)
test_targets[np.arange(test[1].shape[0]), :, test[1]] = 1.0

train = train[0].reshape((-1, 1, 784, 1))
test = test[0].reshape((-1, 1, 784))

lstm_test.main(784, 1, 10, (train, train_targets), (test, test_targets))