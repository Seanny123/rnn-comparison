# for each dataset increase the number of classes and mark how quickly
# performance degrades
# TODO: maybe abstract this to a general experiment function?

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
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

freq_list = [10, 10, 150,]
class_type_list = ["cont_spec", "orth_spec", "disc_spec",]
exp_iter = 5
class_nums = [40]
rc_acc = []
svm_acc = []
rc_diff = []
svm_diff = []

for c_i, cls_type in enumerate(class_type_list):
    for n_classes in class_nums:
        mk_res = mk_cls_dataset(t_len=0.25, dims=1, n_classes=n_classes,
            freq=freq_list[c_i], class_type=cls_type)
        desc = mk_res[1]
        dat = np.array(mk_res[0])
        dat_arg = make_run_args(dat, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt))

        for e_i in range(exp_iter):
            aug_res = aug(dat, mk_res[1], 1, add_rand_noise, {})
            test_arg = make_run_args(np.array(aug_res), desc["dims"], desc["n_classes"], int(desc["t_len"]/dt))


            tmp_res = rc_nengo_test.main(desc["t_len"], desc["dims"], desc["n_classes"], dat_arg, test_arg)
            rc_acc.append(tmp_res[0])
            rc_diff.append(tmp_res[1])
            tmp_res = svm_nengo_test.main(desc["t_len"], desc["dims"], desc["n_classes"], dat_arg, test_arg)
            svm_acc.append(tmp_res[0])
            svm_diff.append(tmp_res[1])
            print("\n\n Finished Iteration %s For Class %s %s Acc: %s, %s\n\n" %(e_i, cls_type, n_classes, rc_acc[-1], svm_acc[-1]))


filename = "results/class_exp_%s" %(datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, rc_res={"acc":rc_acc, "diff":rc_diff}, svm_res={"acc":svm_acc, "diff":svm_diff}, class_nums=class_nums, exp_iter=exp_iter)
ipdb.set_trace()