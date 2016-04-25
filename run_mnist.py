import lstm_test
import pickle
import numpy as np
from dataman import make_run_args
from constants import *

import matplotlib.pyplot as plt

import ipdb

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

with open("mnist.pkl", "rb") as f:
    train, _, test = pickle.load(f)

train_targets = np.zeros((train[1].shape[0], 1, 10), dtype=np.float32)
train_targets[np.arange(train[1].shape[0]), :, train[1]] = 1.0

test_targets = np.zeros((test[1].shape[0], 1, 10), dtype=np.float32)
test_targets[np.arange(test[1].shape[0]), :, test[1]] = 1.0

train = train[0].reshape((-1, 1, 784))
test = test[0]

lstm_test.main(784, 1, 10, (train, train_targets), (test, test_targets))
