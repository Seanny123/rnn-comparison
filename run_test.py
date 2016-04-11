import numpy as np
import ipdb

from constants import *
import rc_nengo_test
from dataman import make_correct

def make_run_args(fi, dims, t_steps):
    """reshape before passing (stop organising by class) 
    and get the correct-ans and pass that too"""
    dat = fi["class_sig_list"]
    cls_num = dat.shape[0]
    sig_num = dat.shape[1]
    return (dat.reshape((int(cls_num*sig_num), dims, t_steps)), make_correct(dat))

# load a dataset for training
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
desc = fi["class_desc"].item()
dat_arg = make_run_args(fi, desc["dims"], int(desc["t_len"]/dt))

# do the same for the test set
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
test_arg = make_run_args(fi, desc["dims"], int(desc["t_len"]/dt))

# run the specific test
rc_nengo_test.main(desc["t_len"], desc["dims"], desc["n_classes"], dat_arg, test_arg)