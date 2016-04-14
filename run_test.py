import numpy as np
import ipdb

from constants import *
import rc_nengo_test
import rnn_test
from dataman import make_correct


def make_run_args(fi, dims, n_classes, t_steps, ann=False):
    """reshape before passing (stop organising by class) 
    and get the correct-ans and pass that too"""
    dat = fi["class_sig_list"]
    cls_num = dat.shape[0]
    sig_num = dat.shape[1]

    if ann:
        pause_size = int(PAUSE/dt)
        ann_shape = int(cls_num*sig_num*(t_steps+pause_size))

        dat = np.concatenate(
            (np.zeros((int(cls_num*sig_num*dims), pause_size)),
            dat.reshape((int(cls_num*sig_num*dims), t_steps))),
            axis=1
        ).reshape((ann_shape, dims))

        cor = make_correct(
            fi["class_sig_list"],
            n_classes
        )
        cor = np.concatenate(
            (np.zeros((n_classes*sig_num, n_classes, pause_size)),
            # stretch it to fit the time slots
            # the reshape for concat
            np.repeat(cor, t_steps, axis=1).reshape(n_classes*sig_num, n_classes, t_steps)),
            axis=2
        ).reshape((-1, t_steps+pause_size))
        return (dat, cor)
    else:
        final_shape = (int(cls_num*sig_num), dims, t_steps)
        return (dat.reshape(final_shape), make_correct(dat, n_classes))


# load a dataset for training
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
desc = fi["class_desc"].item()
dat_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

# do the same for the test set
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
test_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

# run the specific test
#rc_nengo_test.main(desc["t_len"], desc["dims"], desc["n_classes"], dat_arg, test_arg)
rnn_test.main(desc["t_len"]+PAUSE, desc["dims"], desc["n_classes"], dat_arg, test_arg)