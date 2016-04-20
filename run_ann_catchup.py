import numpy as np
from augman import ann_shuffle
from dataman import make_run_args, mk_cls_dataset
import rnn_test
import deep_test
from constants import *
import matplotlib.pyplot as plt
import os

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import ipdb

mk_cls_dataset(t_len=0.5, dims=2, n_classes=4, freq=0, class_type="flat")

# get ANN data
fi = np.load("datasets/dataset_flatcls_0.5_2_4_0.npz")
desc = fi["class_desc"].item()
dat_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

# shuffle the hell out of it
new_arg = ann_shuffle(dat_arg[0], dat_arg[1], desc["t_len"], repeats=50)

# run and plot the results
#rnn_test.main(desc["t_len"], desc["dims"], desc["n_classes"], new_arg, dat_arg)
deep_test.main(desc["t_len"], desc["dims"], desc["n_classes"], new_arg, dat_arg)