import numpy as np
from augman import *
from dataman import make_run_args
import matplotlib.pyplot as plt

import ipdb

# load the data
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
desc = fi["class_desc"].item()
dat_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

# make sure the pattern being returned is reasonable
plt.plot(dat_arg[0][:, 0, 0])
plt.plot(dat_arg[1][:, 0, 0])
plt.show()

# augment it
new_arg = ann_shuffle(dat_arg[0], dat_arg[1], desc["t_len"])

# test that it doesn't suck
plt.plot(new_arg[0][:, 0, 0])
plt.plot(new_arg[1][:, 0, 0])
plt.show()