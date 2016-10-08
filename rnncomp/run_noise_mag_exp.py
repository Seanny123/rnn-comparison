# increase the amplitude of the additive noise
# Bonus: try the shot-noise and lag as well

import random
import numpy as np
from constants import *

np.random.seed(SEED)
random.seed(SEED)

from dataman import *
from augman import *
import rc_nengo_test
import svm_nengo_test

import datetime
import ipdb

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

freq_list = [10, 10, 20,]
class_type_list = ["cont_spec", "orth_spec", "disc_spec",]
exp_iter = 10
n_classes = 10

noise_funcs = [
    add_rand_noise,
    add_rand_noise,
    add_rand_noise,
    add_rand_noise,
]
noise_kw_args = [
    {"scale": 0.0},
    {"scale": 0.2},
    {"scale": 0.4},
    {"scale": 0.6},
]

# detailed results for debugging later saved as numpy archive
rc_pred = []
svm_pred = []
van_pred = []
rc_cor = []
svm_cor = []
van_cor = []

# summary of results for plotting later to be converted into Pandas
pd_columns = ['t_len', 'dims', 'n_classes', 'approach', 'accuracy',
              'ad_mean', 'ad_std', 'gd_mean', 'gd_std',
              'conf_mean', 'conf_std']
pd_res = []

noise_names = []
for n_f in noise_funcs:
    noise_names.append(n_f.__name__)


for c_i, cls_type in enumerate(class_type_list):
    for n_i, noise in enumerate(noise_funcs):
        mk_res = mk_cls_dataset(t_len=0.25, dims=1, n_classes=n_classes,
            freq=freq_list[c_i], class_type=cls_type)
        desc = mk_res[1]
        dat = np.array(mk_res[0])
        dat_arg = make_run_args(dat, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt))

        rc_train, rc_test = rc_nengo_test.reservoir(desc["t_len"], desc["dims"], desc["n_classes"], alif=False)
        rc_model = rc_train(dat_arg)

        svm_train, svm_test = svm_nengo_test.svm_freq(desc["t_len"], desc["dims"], desc["n_classes"], alif=False)
        svm_model = svm_train(dat_arg)

        for e_i in range(exp_iter):
            aug_res = aug(dat, mk_res[1], 1, noise, noise_kw_args[n_i])
            test_arg = make_run_args(np.array(aug_res), desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), shuffle=True)

            test_res = rc_test(*rc_model, testset=test_arg)
            acc_res = get_accuracy(*test_res, t_len=desc["t_len"])
            rc_acc.append(acc_res[0])
            rc_diff.append(acc_res[1])

            test_res = svm_test(*svm_model, testset=test_arg)
            acc_res = get_accuracy(*test_res, t_len=desc["t_len"])
            svm_acc.append(acc_res[0])
            svm_diff.append(acc_res[1])

            print("\n\n Finished Iteration %s For Class %s %s Acc: %s, %s\n\n" %(e_i, cls_type, n_classes, rc_acc[-1], svm_acc[-1]))


filename = "results/noise_exp_%s" %(datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, rc_res={"acc":rc_acc, "diff":rc_diff}, svm_res={"acc":svm_acc, "diff":svm_diff}, noise_funcs=noise_names, noise_args=noise_kw_args, exp_iter=exp_iter)
ipdb.set_trace()