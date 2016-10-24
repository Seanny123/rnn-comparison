from constants import *
from dataman import mk_cls_dataset, make_run_args_ann, make_run_args_nengo
from augman import dat_shuffle, add_rand_noise, dat_repshuf, pre_arggen_repeat, aug
from run_utils import run_van, run_fancy_van, save_results, make_noisy_arg, run_rc

import numpy as np

import datetime
import os

import ipdb
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,cnmem=1000"

freq_list = [10, 10, 20]
class_type_list = ["cont_spec", "orth_spec", "disc_spec"]
exp_iter = 10
n_classes = 5
dup_num = 100
noise_kw_args = {"scale": 0.005, "sig": False}

# detailed results for debugging later saved as numpy archive
res_dict = dict()
res_dict["rc_res"] = {"pred": [], "cor": []}
res_dict["fvan_res"] = {"pred": [], "cor": []}
res_dict["noisy_van_res"] = {"pred": [], "cor": []}
res_dict["shuf_van_res"] = {"pred": [], "cor": []}

# summary of results for plotting later to be converted into Pandas
pd_columns = ['t_len', 'dims', 'n_classes', 'approach', 'accuracy',
              'ad_mean', 'ad_std', 'gd_mean', 'gd_std',
              'conf_mean', 'conf_std', "aug method"]
pd_res = []

desc = dict()

for c_i, cls_type in enumerate(class_type_list):
    mk_res = mk_cls_dataset(t_len=0.5, dims=1, n_classes=n_classes,
                            freq=freq_list[c_i], class_type=cls_type, save_dir="../datasets")
    desc = mk_res[1]
    dat = np.array(mk_res[0])

    make_basic_arg = make_noisy_arg(dat, desc, add_rand_noise, noise_kw_args=noise_kw_args)

    for e_i in range(exp_iter):
        dat_arg, dat_cor, test_arg = make_basic_arg()

        # test data
        shuf_dat = dat_shuffle(*test_arg)
        ann_t_dat, ann_t_cor = make_run_args_ann(*shuf_dat)

        # shuffle only, as a baseline for comparison
        rep_dat, rep_cor = dat_repshuf(dat_arg, dat_cor, reps=dup_num)
        ann_dat, ann_cor = make_run_args_ann(rep_dat, rep_cor)

        # run vRNN
        run_van(res_dict["shuf_van_res"]["pred"], res_dict["shuf_van_res"]["cor"],
                ann_dat, ann_cor, (ann_t_dat, ann_t_cor), desc, pd_res, log_other=["repeat"])

        # shuffle with noise
        rep_noisy = aug(pre_arggen_repeat(dat, dup_num), desc, dup_num,
                        add_rand_noise, kwargs=noise_kw_args)
        rep_dat, rep_cor = make_run_args_nengo(np.array(rep_noisy))
        nengo_dat, nengo_cor = dat_shuffle(rep_dat, rep_cor)

        ann_dat, ann_cor = make_run_args_ann(nengo_dat, nengo_cor)

        # run vRNN
        run_van(res_dict["noisy_van_res"]["pred"], res_dict["noisy_van_res"]["cor"],
                ann_dat, ann_cor, (ann_t_dat, ann_t_cor), desc, pd_res, log_other=["repeat with noise"])

        # run fancy vRNN
        run_fancy_van(res_dict["fvan_res"]["pred"], res_dict["fvan_res"]["cor"],
                      ann_dat, ann_cor, (ann_t_dat, ann_t_cor), desc, pd_res, log_other=["repeat with noise"])

        current_time = datetime.datetime.now().strftime("%I:%M:%S")
        print("\n\n Finished Iteration %s at %s" % (e_i, current_time))
        print("Accuracy shuffle:%s, reservoir:%s, with_noise:%s, fancy_init:%s\n\n"
              % (pd_res[-4][acc_idx], pd_res[-3][acc_idx], pd_res[-2][acc_idx], pd_res[-1][acc_idx]))

    current_time = datetime.datetime.now().strftime("%I:%M:%S")
    print("Finished %s at %s\n" % (cls_type, current_time))

# save raw results
class_desc = dict()
class_desc["t_len"] = desc["t_len"]
class_desc["dims"] = desc["dims"]
class_desc["SEED"] = SEED
class_desc["sample_every"] = sample_every
class_desc["PAUSE"] = PAUSE
class_desc["exp_iter"] = exp_iter

base_name = "ann_catchup_res_%s" % dup_num

save_results(pd_res, pd_columns, res_dict, base_name, class_desc)

ipdb.set_trace()
