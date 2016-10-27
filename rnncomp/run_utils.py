from post import add_to_pd
from augman import aug, dat_shuffle
from dataman import make_run_args_nengo, make_run_args_ann
import rc_nengo_test
import svm_nengo_test
import rnn_test
from constants import *

import numpy as np
import pandas as pd
import lasagne

import datetime
import ipdb


def run_rc(rc_pred, rc_cor, dat_arg, dat_cor, test_arg, desc, pd_res, log_other=list(), alif=False):
    rc_train, rc_test = rc_nengo_test.reservoir(desc["t_len"], desc["dims"], desc["n_classes"], alif)
    test_model = rc_train(dat_arg, dat_cor)
    tmp_res = rc_test(*test_model, testset=test_arg)
    rc_pred.append(tmp_res[0])
    rc_cor.append(tmp_res[1])

    add_to_pd(pd_res, desc, "RC", rc_pred[-1], rc_cor[-1], sample_every, log_other)


def run_svm(svm_pred, svm_cor, dat_arg, dat_cor, test_arg, desc, pd_res, log_other=list(), alif=False):
    svm_train, svm_test = svm_nengo_test.svm_freq(desc["t_len"], desc["dims"], desc["n_classes"], alif)
    test_model = svm_train(dat_arg, dat_cor)
    tmp_res = svm_test(*test_model, testset=test_arg)
    svm_pred.append(tmp_res[0])
    svm_cor.append(tmp_res[1])

    add_to_pd(pd_res, desc, "SVM", svm_pred[-1], svm_cor[-1], sample_every, log_other)


def run_van(van_pred, van_cor, dat_arg, dat_cor, test_arg, desc, pd_res, log_other):
    van_train, van_test = rnn_test.vanilla(desc["dims"], desc["n_classes"])
    van_sim, p_out = van_train(dat_arg, dat_cor)
    van_pred.append(van_test(van_sim, test_arg[0], p_out))
    van_cor.append(test_arg[1])

    add_to_pd(pd_res, desc, "vRNN", van_pred[-1], van_cor[-1], sample_every, log_other)


def run_fancy_van(van_pred, van_cor, dat_arg, dat_cor, test_arg, desc, pd_res, log_other):
    van_train, van_test = rnn_test.vanilla(desc["dims"], desc["n_classes"])
    van_sim, p_out = van_train(dat_arg, dat_cor, w_rec_init=rnn_test.Ident,
                               nonlin=lasagne.nonlinearities.rectify, learning_rate=0.001)
    van_pred.append(van_test(van_sim, test_arg[0], p_out))
    van_cor.append(test_arg[1])

    add_to_pd(pd_res, desc, "fvRNN", van_pred[-1], van_cor[-1], sample_every, log_other)


def make_noisy_arg(dat, desc, noise_func=None, noise_kw_args=None):

    def f():
        aug_res = aug(dat, desc, 1, noise_func, noise_kw_args)
        dat_arg, dat_cor = make_run_args_nengo(np.array(aug_res))

        aug_res = aug(dat, desc, 1, noise_func, noise_kw_args)
        t_dat_arg, t_dat_cor = make_run_args_nengo(np.array(aug_res))
        test_arg = dat_shuffle(t_dat_arg, t_dat_cor)

        return dat_arg, dat_cor, test_arg

    return f


def make_plain_arg(dat, desc, noise_func=None, noise_kw_args=None):
    def f():
        aug_res = aug(dat, desc, 1, noise_func, noise_kw_args)
        dat_arg, dat_cor = make_run_args_nengo(np.array(aug_res))
        test_arg = dat_shuffle(dat_arg, dat_cor)
        return dat_arg, dat_cor, test_arg

    return f


def run_alif_exp(desc, exp_iter, pd_res, res_dict, make_arg_func):

    for e_i in range(exp_iter):

        dat_arg, dat_cor, test_arg = make_arg_func()

        # run Nengo nets only
        run_rc(res_dict["rc_res"]["pred"], res_dict["rc_res"]["cor"],
               dat_arg, dat_cor, test_arg, desc, pd_res, alif=True)
        run_svm(res_dict["svm_res"]["pred"], res_dict["svm_res"]["cor"],
                dat_arg, dat_cor, test_arg, desc, pd_res, alif=True)

        current_time = datetime.datetime.now().strftime("%I:%M:%S")
        print("\n\n Finished Iteration %s at %s" % (e_i, current_time))
        print("Accuracy RC:%s, SVM:%s\n\n" % (pd_res[-2][acc_idx], pd_res[-1][acc_idx]))


def run_exp(desc, exp_iter, pd_res, res_dict, make_arg_func, log_other=list()):

    for e_i in range(exp_iter):

        dat_arg, dat_cor, test_arg = make_arg_func()

        # run Nengo nets
        run_rc(res_dict["rc_res"]["pred"], res_dict["rc_res"]["cor"],
               dat_arg, dat_cor, test_arg, desc, pd_res, log_other)
        run_svm(res_dict["svm_res"]["pred"], res_dict["svm_res"]["cor"],
                dat_arg, dat_cor, test_arg, desc, pd_res, log_other)

        # prep vRNN
        ann_dat, ann_cor = make_run_args_ann(dat_arg, dat_cor)
        shuf_dat = dat_shuffle(*test_arg)
        ann_t_dat, ann_t_cor = make_run_args_ann(*shuf_dat)

        # run vRNN
        run_van(res_dict["van_res"]["pred"], res_dict["van_res"]["cor"],
                ann_dat, ann_cor, (ann_t_dat, ann_t_cor), desc, pd_res, log_other)

        current_time = datetime.datetime.now().strftime("%I:%M:%S")
        print("\n\n Finished Iteration %s at %s" % (e_i, current_time))
        print("Accuracy RC:%s, SVM:%s, vRNN:%s\n\n"
              % (pd_res[-3][acc_idx], pd_res[-2][acc_idx], pd_res[-1][acc_idx]))


def save_results(pd_res, pd_columns, res_dict, base_name="", class_desc=None):
    class_desc = class_desc or {}

    filename = "../results/%s_%s" % (base_name, datetime.datetime.now().strftime("%I_%M_%S"))
    np.savez(filename, res_dict["rc_res"], res_dict["svm_res"], res_dict["van_res"], class_desc=class_desc)

    ipdb.set_trace()
    # save processed results
    df = pd.DataFrame(pd_res, columns=pd_columns)
    hdf = pd.HDFStore("../results/%s_%s.h5" % (base_name, datetime.datetime.now().strftime("%I_%M_%S")))
    df.to_hdf(hdf, base_name)
