import numpy as np
from augman import ann_shuffle
from dataman import make_run_args, mk_cls_dataset
import matplotlib.pyplot as plt
from constants import *
from collections import defaultdict

import ipdb

#mk_res = mk_cls_dataset(t_len=0.5, dims=2, n_classes=4, freq=0, class_type="flat")

# load the data
fi = np.load("datasets/dataset_flatcls_0.5_2_4_0.npz")

desc = fi["class_desc"].item()
dat_arg = make_run_args(fi, desc["dims"], desc["n_classes"], int(desc["t_len"]/dt), ann=True)

#ipdb.set_trace()
tmp = defaultdict(list)
for a, b in zip(dat_arg[0][:, 0], dat_arg[1][:, 0]):
    a = tuple(a)
    b = tuple(b)
    if b not in tmp[a]:
        tmp[a] += [b]
print(tmp)

'''
# make sure the pattern being returned is reasonable
assert dat_arg[1].shape[2] == desc["n_classes"]
plt.plot(dat_arg[0][:, 0, 0])
for i in range(desc["n_classes"]):
    plt.plot(dat_arg[1][:, 0, i])
plt.ylim(-0.1, 1.1)
plt.show()

# augment it
new_arg = ann_shuffle(dat_arg[0], dat_arg[1], desc["t_len"])

# test that it doesn't suck
assert new_arg[1].shape[2] == desc["n_classes"]
plt.plot(new_arg[0][:, 0, 0])
for i in range(desc["n_classes"]):
    plt.plot(new_arg[1][:, 0, i])
plt.ylim(-0.1, 1.1)
plt.show()
'''
