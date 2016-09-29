# run the most basic test to make sure the networks are working correctly
import datetime

import ipdb

import rnn_test
from dataman import *
from augman import dat_shuffle


# load a dataset for training
fi = np.load("../datasets/dataset_flatcls_0.5_1_3_0.npz")
desc = fi["class_desc"].item()
nengo_dat, nengo_cor = make_run_args_nengo(fi)
ann_dat, ann_cor = make_run_args_ann(nengo_dat, nengo_cor)

# make test set
shuf_dat = dat_shuffle(nengo_dat, nengo_cor)
test_dat, test_cor = make_run_args_ann(*shuf_dat)

van_train, van_test = rnn_test.vanilla(desc["dims"], desc["n_classes"])
van_sim, p_out = van_train(ann_dat, ann_cor)
res = van_test(van_sim, test_dat, p_out)

filename = "../results/basic_ann_exp_%s" %(datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, van_res={"pred": res, "cor": test_cor})

ipdb.set_trace()
