# Different augmentation functions
import nengo
from nengo.processes import WhiteSignal, WhiteNoise
from dataman import d3_scale
from constants import *
import numpy as np
import json

import matplotlib.pyplot as plt
import ipdb


def aug(dataset, desc, amount, func, kwargs):
    """iterate through each class, augment it and save the result"""
    new_data = []

    for _ in range(desc["n_classes"]):
        new_data.append(
            np.zeros((amount, desc["dims"], int(desc["t_len"]/dt)))
        )

    # TODO: tell these for-loops to calm the hell down
    for c_i in range(len(dataset)):
        for s_i in xrange(amount):
            for d_i in xrange(desc["dims"]):
                new_data[c_i][s_i, d_i] = func(dataset[c_i][0][d_i], desc["t_len"], **kwargs)

    filename = "../datasets/dataset_%scls_%s_%s_%s_%s_aug_%s" %(
        desc["class_type"],
        desc["t_len"],
        desc["dims"],
        desc["n_classes"],
        desc["SEED"],
        func.__name__
    )

    # Maintain a list of all the augmented files using json
    with open("../datasets/aug_log.json", "r") as f_log:
        file_list = json.loads(f_log.read())
    file_list[filename] = {"desc":desc, "func": func.__name__, "kwargs":kwargs}
    with open("../datasets/aug_log.json", "w") as f_log:
        f_log.write(json.dumps(file_list))

    # TODO: give option to append to an existing file if the filename already exists
    # especially with the same kwargs
    np.savez(filename, class_sig_list=new_data, class_desc=desc)

    return new_data

# TODO: try gaussian and uniform
def add_rand_noise(dataset, t_len, freq=10, scale=0.2, sig=True):
    """additive noise"""
    if sig:
        noise = WhiteSignal(t_len, freq).run(t_len)[:, 0] * scale
    else:
        noise = WhiteNoise().run(t_len)[:, 0] * scale
    return dataset + noise

def conv_rand_noise(dataset, t_len, freq=500, scale=0.001):
    """convolve with noise"""
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
    #ipdb.set_trace()
    dataset.reshape((-1, width))[shot_ind, :] += shot_vals
    # no need to rescale
    return dataset.flatten()

"""
 filtering a signal doesn't need to be a function, because we have no
 idea what we're going to be filtering with, other than a LowPass for lolz
"""

def offset(dataset, scale=0.1):
    """shift and normalize"""
    return d3_scale(dataset + scale)


def make_more(dataset):
    # won't work for '_spec' signals
    raise NotImplementedError("Nope")

def lag(dataset, t_len, lags=3, width=10):
    """add a random temporal lag"""
    t_len = t_len/dt
    assert lags*width < t_len
    assert t_len % width == 0
    lag_ind = np.random.choice(int(t_len/width), replace=False, size=lags)

    # reshape into widths, index to add then flatten again
    lag_vals = dataset.reshape((-1, width))[lag_ind, 0]
    dataset.reshape((-1, width))[lag_ind, :] = (np.ones((width, lags)) * lag_vals).T
    # no need to rescale
    return dataset.flatten()

def ann_repeat(dat, cor, t_len, repeats=3, rng=np.random.RandomState(SEED)):
    """because online training is weird, shuffle the presentation order
    of the signal, but repeat the dataset"""
    sig_len = int((t_len + PAUSE)/dt)
    sig_num = dat.shape[0]/sig_len
    rp_len = dat.shape[0]
    ind = np.arange(sig_num)

    # this repition needs to be reshaped after repeating, maybe should use tile instead?
    final_dat = np.tile(dat, (repeats, 1, 1))
    final_cor = np.tile(cor, (repeats, 1, 1))
    for r_i in xrange(1, repeats):
        rng.shuffle(ind)
        dat_chunk = final_dat[r_i*rp_len:(r_i+1)*rp_len].reshape((sig_num, sig_len, 1, -1))
        dat_chunk = dat_chunk[ind]
        final_dat[r_i*rp_len:(r_i+1)*rp_len] = dat_chunk.reshape((rp_len, 1, -1))

        cor_chunk = final_cor[r_i*rp_len:(r_i+1)*rp_len].reshape((sig_num, sig_len, 1, -1))
        cor_chunk = cor_chunk[ind]
        final_cor[r_i*rp_len:(r_i+1)*rp_len] = cor_chunk.reshape((rp_len, 1, -1))

    return (final_dat, final_cor)

def dat_shuffle(dat, cor, rng=np.random.RandomState(SEED)):
    idx = np.arange(cor.shape[0])
    rng.shuffle(idx)
    cor = cor[idx]
    dat = dat[idx]
    return (dat, cor)