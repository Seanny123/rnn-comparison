import numpy as np
import ipdb

from constants import *
import rc_nengo_test
import svm_nengo_test
import rnn_test
from dataman import make_correct, make_run_args
from augman import ann_shuffle, nengo_shuffle

import matplotlib.pyplot as plt


# load a dataset for training
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
desc = fi["class_desc"].item()
dat_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt))
ann_dat = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)
ann_dat = an

# do the same for the test set
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
test_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), shuffle=True)
ann_test = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), shuffle=True, ann=True)

# run the specific test
"""
rc_train, rc_test = rc_nengo_test.reservoir(desc["t_len"], desc["dims"], desc["n_classes"])
test_model = rc_train(dat_arg)
res = rc_test(*test_model, testset=test_arg)
plt.plot(res[0])
plt.plot(res[1])
plt.show()

svm_train, svm_test = svm_nengo_test.svm_freq(desc["t_len"], desc["dims"], desc["n_classes"])
test_model = svm_train(dat_arg)
res = svm_test(*test_model, testset=test_arg)
plt.plot(res[0])
plt.plot(res[1])
plt.show()
"""

van_train, van_test = rnn_test.vanilla(desc["t_len"], desc["dims"], desc["n_classes"])
van_sim = van_train(ann_dat)
res = van_test(van_sim, ann_test)
plt.plot(res[0])
plt.plot(res[1])
plt.show()

ipdb.set_trace()
