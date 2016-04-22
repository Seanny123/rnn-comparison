# increase the amplitude of the additive noise
# Bonus: try the shot-noise and lag as well

import random
import numpy as np
from constants import *

np.random.seed(SEED)
random.seed(SEED)

from dataman import *
from augman import *
import rnn_test

import datetime
import ipdb

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

freq_list = [10, 10, 150,]
class_type_list = ["cont_spec", "orth_spec", "disc_spec",]
exp_iter = 10
n_classes = 4
ann_acc = []
ann_diff = []
noise_funcs = [
    add_rand_noise,
    add_rand_noise,
    add_rand_noise,
]

noise_kw_args = [
    {"scale":0.005, "sig":False},
    {"scale":0.01, "sig":False},
    {"scale":0.02, "sig":False},
]

noise_names = []
for n_f in noise_funcs:
    noise_names.append(n_f.__name__)

# figure out how much noise is needed before degredation
# then run it
# then try combining noise

for c_i, cls_type in enumerate(class_type_list):
    for n_i, noise in enumerate(noise_funcs):
        mk_res = mk_cls_dataset(t_len=0.25, dims=1, n_classes=n_classes,
            freq=freq_list[c_i], class_type=cls_type)
        desc = mk_res[1]
        dat = np.array(aug(np.array(mk_res[0]), mk_res[1], 1, noise, noise_kw_args[n_i]))
        dat_arg = make_run_args(dat, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

        for e_i in range(exp_iter):
            aug_res = aug(dat, mk_res[1], 1, noise, noise_kw_args[n_i])
            test_arg = make_run_args(np.array(aug_res), desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

            tmp_res = rnn_test.main(desc["t_len"], desc["dims"], desc["n_classes"], dat_arg, test_arg)
            ann_acc.append(tmp_res[0])
            ann_diff.append(tmp_res[1])
            print("\n\n Finished Iteration %s For Class %s %s Acc: %s\n\n" %(e_i, cls_type, n_classes, ann_acc[-1]))

filename = "results/ann_noise_exp_%s" %(datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, ann_res={"acc":ann_acc, "diff":ann_diff}, noise_funcs=noise_names, noise_args=noise_kw_args, exp_iter=exp_iter)
ipdb.set_trace()