# for each dataset try using ALIF neurons and check if there's a performance
# boost
# TODO: maybe abstract this to a general experiment function?

import random
import numpy as np
import pandas as pd
from constants import *

np.random.seed(SEED)
random.seed(SEED)

from dataman import *
from augman import *
from post import get_accuracy
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
exp_iter = 2
class_nums = [10, 20, 30,]
rc_acc = []
svm_acc = []
rc_diff = []
svm_diff = []

res_frame = []
frame_columns = ["Classifier Type", "Signal Type", "Accuracy", "Number of Classes", "Neuron Type",]

alif = False
if alif:
    n_type = "Adaptive LIF"
else:
    n_type = "LIF"

for c_i, cls_type in enumerate(class_type_list):
    for n_classes in class_nums:
        mk_res = mk_cls_dataset(t_len=0.25, dims=1, n_classes=n_classes,
            freq=freq_list[c_i], class_type=cls_type)
        desc = mk_res[1]
        dat = np.array(mk_res[0])
        dat_arg = make_run_args(dat, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt))

        rc_train, rc_test = rc_nengo_test.reservoir(desc["t_len"], desc["dims"], desc["n_classes"], alif=alif)
        rc_model = rc_train(dat_arg)

        svm_train, svm_test = svm_nengo_test.svm_freq(desc["t_len"], desc["dims"], desc["n_classes"], alif=alif)
        svm_model = svm_train(dat_arg)

        for e_i in range(exp_iter):
            aug_res = aug(dat, mk_res[1], 1, add_rand_noise, {})
            # TODO: fix the shuffle
            test_arg = make_run_args(np.array(aug_res), desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), shuffle=True)

            test_res = rc_test(*rc_model, testset=test_arg)
            acc_res = get_accuracy(*test_res, t_len=desc["t_len"])
            rc_acc.append(acc_res[0])
            rc_diff.append(acc_res[1])
            res_frame.append(("SVM", cls_type, acc_res[0], n_classes, n_type))

            test_res = svm_test(*svm_model, testset=test_arg)
            acc_res = get_accuracy(*test_res, t_len=desc["t_len"])
            svm_acc.append(acc_res[0])
            svm_diff.append(acc_res[1])
            res_frame.append(("RC", cls_type, acc_res[0], n_classes, n_type))

            print("\n\n Finished Iteration %s For Class %s %s Acc: %s, %s\n\n" %(e_i, cls_type, n_classes, rc_acc[-1], svm_acc[-1]))


filename = "results/alif_exp_%s" %(datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, rc_res={"acc":rc_acc, "diff":rc_diff}, svm_res={"acc":svm_acc, "diff":svm_diff}, class_nums=class_nums, exp_iter=exp_iter, alif=alif)
dat_frame = pd.DataFrame(res_frame, columns=frame_columns)
dat_frame.to_csv(filename + ".csv")
ipdb.set_trace()