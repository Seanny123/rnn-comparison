# run the most basic test to make sure the networks are working correctly
import rc_nengo_test
import svm_nengo_test
from dataman import *
from augman import dat_shuffle

import datetime
import ipdb


# load a dataset for training
fi = np.load("../datasets/dataset_flatcls_0.5_1_3_0.npz")
desc = fi["class_desc"].item()
dat_arg, dat_cor = make_run_args_nengo(fi)

# make test set
test_arg = dat_shuffle(dat_arg, dat_cor)

# run the specific test
rc_train, rc_test = rc_nengo_test.reservoir(desc["t_len"], desc["dims"], desc["n_classes"])
test_model = rc_train(dat_arg, dat_cor)
rc_pred, rc_cor = rc_test(*test_model, testset=test_arg)

svm_train, svm_test = svm_nengo_test.svm_freq(desc["t_len"], desc["dims"], desc["n_classes"])
test_model = svm_train(dat_arg, dat_cor)
svm_pred, svm_cor = svm_test(*test_model, testset=test_arg)

filename = "../results/basic_exp_%s" %(datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, rc_res={"pred": rc_pred, "cor": rc_cor}, svm_res={"pred": svm_pred, "cor": svm_cor},)

ipdb.set_trace()
