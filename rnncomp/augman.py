# Different augmentation functions
import nengo
from nengo.processes import WhiteSignal, WhiteNoise
from dataman import d3_scale
from constants import *
import numpy as np

import json
import itertools

import matplotlib.pyplot as plt
import ipdb


# TODO: get rid of the amount argument
def aug(dataset, desc, amount, func, kwargs=None, save_dir="../datasets"):
    """iterate through each class, augment it and save the result.

    The `amount` argument is to note the number of examples"""
    kwargs = kwargs or {}
    new_data = []

    for _ in range(desc["n_classes"]):
        new_data.append(
            np.zeros((amount, desc["dims"], int(desc["t_len"]/dt)))
        )

    for c_i, s_i, d_i in itertools.product(xrange(len(dataset)), xrange(amount), xrange(desc["dims"])):
        new_data[c_i][s_i, d_i] = func(dataset[c_i][s_i][d_i], desc["t_len"], **kwargs)

    if save_dir is not None:
        filename = "%s/dataset_%scls_%s_%s_%s_%s_aug_%s" % (
            save_dir,
            desc["class_type"],
            desc["t_len"],
            desc["dims"],
            desc["n_classes"],
            desc["SEED"],
            func.__name__
        )

        # Maintain a list of all the augmented files using json
        with open("%s/aug_log.json" % save_dir, "r") as f_log:
            file_list = json.loads(f_log.read())
        file_list[filename] = {"desc": desc, "func": func.__name__, "kwargs": kwargs}
        with open("%s/aug_log.json" % save_dir, "w") as f_log:
            f_log.write(json.dumps(file_list))

        # TODO: give option to append to an existing file if the filename already exists
        # especially with the same kwargs
        np.savez(filename, class_sig_list=new_data, class_desc=desc)

    return new_data


def add_rand_noise(dataset, t_len, freq=10, scale=0.2, sig=True):
    """additive noise"""
    if sig:
        noise = WhiteSignal(t_len, freq).run(t_len)[:, 0] * scale
    else:
        noise = WhiteNoise().run(t_len)[:, 0] * scale
    return dataset + noise


def conv_rand_noise(dataset, t_len, freq=500, scale=0.001):
    """convolve with noise, which heavily distorts the signal almost beyond all recognition"""
    noise = WhiteSignal(t_len, freq).run(t_len)[:, 0] * scale
    return d3_scale(np.convolve(dataset, noise, mode="same"))


def shot_noise(dataset, t_len, shots=3, width=2):
    """randomly add impulses in either direction"""
    t_len = t_len/dt
    assert shots*width < t_len
    assert t_len % width == 0
    shot_ind = np.random.choice(int(t_len/width), replace=False, size=shots)
    shot_vals = np.random.choice([-1, 1], size=(shots, 1))

    # reshape into widths, index to add then flatten again
    ret_val = np.copy(dataset)
    ret_val.reshape((-1, width))[shot_ind, :] += shot_vals
    # no need to rescale
    return ret_val.flatten()


def low_filt(dataset, t_len, tau=0.01):
    return nengo.Lowpass(tau).filtfilt(dataset)


def offset(dataset, t_len, scale=0.1):
    """shift and normalize"""
    if scale > 0.1:
        out_range = (np.min(dataset), np.max(dataset+scale))
    else:
        out_range = (np.min(dataset+scale), np.max(dataset))

    return d3_scale(dataset + scale, out_range=out_range)


def make_more(dataset):
    # won't work for '_spec' signals
    raise NotImplementedError("Nope")


def lag(dataset, t_len, lags=3, width=10):
    """add a random temporal lag"""

    raise NotImplementedError("Nope")
    t_len = t_len/dt
    assert lags*width < t_len
    assert t_len % width == 0
    lag_ind = np.random.choice(int(t_len/width), replace=False, size=lags)
    ret_val = np.copy(dataset)

    # reshape into widths, index to add then flatten again
    lag_vals = ret_val.reshape((-1, width))[lag_ind, 0]
    ret_val.reshape((-1, width))[lag_ind, :] = (np.ones((width, lags)) * lag_vals).T
    # no need to rescale
    return ret_val.flatten()


def pre_arggen_repeat(dat, repeats=3):
    """Repeat the dataset

    Expects data from a created dataset unformatted for specific network input"""

    final_dat = np.tile(dat, (1, repeats, 1, 1))
    return final_dat


def post_arggen_repeat(dat, cor, repeats=3):
    """Repeat the dataset

    Expects data already formatted for Nengo input"""

    final_dat = np.tile(dat, (repeats, 1, 1))
    final_cor = np.tile(cor, (repeats, 1, 1))

    return final_dat, final_cor


def dat_shuffle(dat, cor, rng=np.random.RandomState(SEED)):
    idx = np.arange(cor.shape[0])
    rng.shuffle(idx)
    return dat[idx], cor[idx]


def dat_repshuf(dat, cor, reps=3, rng=np.random.RandomState(SEED)):
    r_dat, r_cor = post_arggen_repeat(dat, cor, reps)
    return dat_shuffle(r_dat, r_cor, rng)

