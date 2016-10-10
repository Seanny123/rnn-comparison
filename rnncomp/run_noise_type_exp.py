# vary the types of noise

from constants import *
from dataman import mk_cls_dataset
from augman import add_rand_noise, shot_noise, offset, low_filt
from run_utils import run_exp, save_results, make_noisy_arg

import numpy as np

import ipdb
import datetime

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


freq_list = [10, 10, 20]
class_type_list = ["cont_spec", "orth_spec", "disc_spec"]
exp_iter = 10
n_classes = 10

noise_funcs = [
    add_rand_noise,
    shot_noise,
    offset,
    low_filt,
]

# detailed results for debugging later saved as numpy archive
res_dict = dict()
res_dict["rc_res"] = {"pred": [], "cor": []}
res_dict["svm_res"] = {"pred": [], "cor": []}
res_dict["van_res"] = {"pred": [], "cor": []}

# summary of results for plotting later to be converted into Pandas
pd_columns = ['t_len', 'dims', 'n_classes', 'approach', 'accuracy',
              'ad_mean', 'ad_std', 'gd_mean', 'gd_std',
              'conf_mean', 'conf_std', "noise type"]
pd_res = []

desc = dict()

noise_names = [n_f.__name__ for n_f in noise_funcs]

for n_i, noise_f in enumerate(noise_funcs):
    for c_i, cls_type in enumerate(class_type_list):
        mk_res = mk_cls_dataset(t_len=0.5, dims=1, n_classes=n_classes,
                                freq=freq_list[c_i], class_type=cls_type, save_dir="../datasets")
        desc = mk_res[1]
        dat = np.array(mk_res[0])

        make_f = make_noisy_arg(dat, desc, noise_f)

        run_exp(desc, exp_iter, pd_res, res_dict, make_f,
                log_other=[noise_names[n_i]])
        current_time = datetime.datetime.now().strftime("%I:%M:%S")
        print("Finished %s at %s" % (cls_type, current_time))


# save raw results
class_desc = dict()
class_desc["t_len"] = desc["t_len"]
class_desc["dims"] = desc["dims"]
class_desc["SEED"] = SEED
class_desc["sample_every"] = sample_every
class_desc["PAUSE"] = PAUSE
class_desc["exp_iter"] = exp_iter

base_name = "noise_type_exp_res"

save_results(pd_res, pd_columns, res_dict, base_name, class_desc)

ipdb.set_trace()
