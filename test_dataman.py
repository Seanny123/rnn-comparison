import random
import numpy as np
from constants import *

np.random.seed(SEED)
random.seed(SEED)

from dataman import *
from augman import *
import ipdb

# make some classes
freq_list = [10, 10, 150, 0,]
class_type_list = ["cont_spec", "orth_spec", "disc_spec", "flat"]
for i in range(4):
    mk_res = mk_cls_dataset(t_len=0.5, dims=2, n_classes=3, freq=freq_list[i], class_type=class_type_list[i])
    aug_res = aug(mk_res[0], mk_res[1], 1, add_rand_noise, {})

# iterate through each class, augment it and save the result

# get some basic training results so if shit hits the fan you have something
# to write about, no matter how shit-tastic it is