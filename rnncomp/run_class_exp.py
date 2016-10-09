# for each dataset increase the number of classes and mark how quickly
# performance degrades given noisy signals and varying the seed of the noise
# Note: training and testing on different noise sigs

from constants import *
from dataman import mk_cls_dataset, make_run_args_nengo, make_run_args_ann
from augman import aug, dat_shuffle, add_rand_noise
import rc_nengo_test
import svm_nengo_test
import rnn_test
from post import add_to_pd

import numpy as np
import pandas as pd

import datetime
import ipdb

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

freq_list = [10, 10, 20]
class_type_list = ["cont_spec", "orth_spec", "disc_spec"]
exp_iter = 10
class_nums = [3, 5, 10, 20, 40]

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

desc = dict()

noise_args = {"sig": False, "scale": 0.1}
acc_idx = 4

for c_i, cls_type in enumerate(class_type_list):
    for n_classes in class_nums:
        mk_res = mk_cls_dataset(t_len=0.5, dims=1, n_classes=n_classes,
                                freq=freq_list[c_i], class_type=cls_type, save_dir="../datasets")
        desc = mk_res[1]
        dat = np.array(mk_res[0])

        for e_i in range(exp_iter):
            # prep Nengo nets
            aug_res = aug(dat, mk_res[1], 1, add_rand_noise, noise_args)
            dat_arg, dat_cor = make_run_args_nengo(np.array(aug_res))
            test_arg = dat_shuffle(dat_arg, dat_cor)

            # run Nengo nets
            rc_train, rc_test = rc_nengo_test.reservoir(desc["t_len"], desc["dims"], desc["n_classes"])
            test_model = rc_train(dat_arg, dat_cor)
            tmp_res = rc_test(*test_model, testset=test_arg)
            rc_pred.append(tmp_res[0])
            rc_cor.append(tmp_res[1])

            add_to_pd(pd_res, desc, "RC", rc_pred[-1], rc_cor[-1], sample_every)

            svm_train, svm_test = svm_nengo_test.svm_freq(desc["t_len"], desc["dims"], desc["n_classes"])
            test_model = svm_train(dat_arg, dat_cor)
            tmp_res = svm_test(*test_model, testset=test_arg)
            svm_pred.append(tmp_res[0])
            svm_cor.append(tmp_res[1])

            add_to_pd(pd_res, desc, "SVM", svm_pred[-1], svm_cor[-1], sample_every)

            # prep vRNN
            ann_dat, ann_cor = make_run_args_ann(dat_arg, dat_cor)
            shuf_dat = dat_shuffle(*test_arg)
            ann_t_dat, ann_t_cor = make_run_args_ann(*shuf_dat)

            # run vRNN
            van_train, van_test = rnn_test.vanilla(desc["dims"], desc["n_classes"])
            van_sim, p_out = van_train(ann_dat, ann_cor)
            van_pred.append(van_test(van_sim, ann_t_dat, p_out))
            van_cor.append(ann_t_cor)
            # that's not the right shape!

            add_to_pd(pd_res, desc, "vRNN", van_pred[-1], van_cor[-1], sample_every)

            current_time = datetime.datetime.now().strftime("%I:%M:%S")
            print("\n\n Finished Iteration %s For Class %s %s at %s" % (e_i, cls_type, n_classes, current_time))
            print("Accuracy RC:%s, SVM:%s, vRNN:%s\n\n"
                  % (pd_res[-3][acc_idx], pd_res[-2][acc_idx], pd_res[-1][acc_idx]))


# save raw results
class_desc = dict()
class_desc["t_len"] = desc["t_len"]
class_desc["dims"] = desc["dims"]
class_desc["SEED"] = SEED
class_desc["sample_every"] = sample_every
class_desc["PAUSE"] = PAUSE
class_desc["exp_iter"] = exp_iter

filename = "../results/class_exp_res_%s" % (datetime.datetime.now().strftime("%I_%M_%S"))
np.savez(filename,
         rc_res={"pred": rc_pred, "cor": rc_cor},
         svm_res={"pred": svm_pred, "cor": svm_cor},
         van_res={"pred": van_pred, "cor": van_cor},
         class_desc=class_desc)

# save processed results
df = pd.DataFrame(pd_res, columns=pd_columns)
hdf = pd.HDFStore("../results/class_exp_res_%s.h5" % (datetime.datetime.now().strftime("%I_%M_%S")))
df.to_hdf(hdf, 'class_exp_res')
ipdb.set_trace()
