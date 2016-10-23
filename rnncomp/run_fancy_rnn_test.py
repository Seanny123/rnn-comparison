# run the fancy test to make sure the networks are working correctly
import rnn_test
from dataman import *
from augman import dat_shuffle

import lasagne

import datetime
import ipdb

# load a dataset for training
fi = np.load("../datasets/dataset_flatcls_0.5_1_4_0.npz")
desc = fi["class_desc"].item()
nengo_dat, nengo_cor = make_run_args_nengo(fi)


# make test set
shuf_dat = dat_shuffle(nengo_dat, nengo_cor)
shuf_again = dat_shuffle(*shuf_dat)

ann_dat, ann_cor = make_run_args_ann(*shuf_dat)
test_dat, test_cor = make_run_args_ann(*shuf_again)

van_train, van_test = rnn_test.vanilla(desc["dims"], desc["n_classes"])
van_sim, p_out = van_train(ann_dat, ann_cor, w_rec_init=rnn_test.Ident, nonlin=lasagne.nonlinearities.rectify)
res = van_test(van_sim, test_dat, p_out)
ipdb.set_trace()

filename = "../results/fancy_ann_exp_%s" % (datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename, van_res={"pred": res, "cor": test_cor})
