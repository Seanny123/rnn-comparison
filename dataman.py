import nengo
import numpy as np
from nengo.processes import WhiteSignal
from constants import *
from random import shuffle
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import ipdb

class_type_list = ["cont_spec", "orth_spec", "disc_spec", "cont_packet",
                   "cont_freq", "cont_amp"]

def d3_scale(dat, out_range=(-1, 1), in_range=None):
    if in_range == None:
        domain = [np.min(dat, axis=0), np.max(dat, axis=0)]
    else:
        domain = in_range

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))

def ortho_nearest(d):
    p = nengo.dists.UniformHypersphere(surface=True).sample(d, d)
    return np.dot(p, np.linalg.inv(sqrtm(np.dot(p.T, p))))

def mk_cls_dataset(t_len, dims, n_classes=2, freq=10, class_type="cont_spec"):
    """given length t_len, dimensions dim, make number of classes given 
    n_classes in terms of a specific signal"""

    assert n_classes >= 2
    assert dims >= 1
    assert t_len > 0

    class_sig_list = []
    # not used for for '_spec' signals
    class_desc = {}
    for n_i in range(n_classes):
        sig = []

        for d_i in range(dims):

            if class_type is "cont_spec":
                """classify using the specific white noise signal"""
                sig.append(
                    d3_scale(
                        WhiteSignal(
                            t_len, freq, seed=(d_i+n_i*n_classes)
                        ).run(t_len)[:, 0]
                    )
                )

            elif class_type is "orth_spec":
                """classify using the specific orthgonal weird signal"""
                assert freq > dims
                vecs = ortho_nearest(freq)[:dims]
                vecs[:, 0] = 0
                vecs[:, -1] = 0
                v_range = np.linspace(0, t_len, freq)
                t_range = np.arange(0, t_len, dt)

                sig = interp1d(v_range, vecs, kind="cubic")(t_range)
                break

            elif class_type is "disc_spec":
                """classify using a specific discrete white-noise
                based signal"""
                assert t_len/dt > freq
                white_size = int(t_len/dt/freq)
                white_vals = np.random.uniform(low=-1, high=1, size=white_size)
                white_noise = np.zeros(int(t_len/dt))
                # TODO: do this without a for-loop by using reshape
                for w_i in xrange(white_vals.shape[0]):
                    white_noise[w_i*freq:(w_i+1)*freq] = white_vals[w_i]

                sig.append(white_noise)

            elif class_type is "cont_packet":
                """classify using a mix of FM and AM packets"""
                packet_length = 3
                assert packet_length % 2 == 1
                # choose a central freq
                # choose a packet diff
                # gen whitenoise signals

                class_desc.append(frs)
                raise NotImplementedError("Nope")

            elif class_type is "cont_freq":
                """classify using the frequency of the white noise signal"""
                # choose from freq range based on n_classes
                class_desc.append(fr)
                raise NotImplementedError("Nope")

            elif class_type is "cont_amp":
                """classify using the amplitude of the white noise signal"""
                # choose from amp range based on n_classes
                class_desc.append(amp)
                raise NotImplementedError("Nope")

            else:
                raise TypeError("Unknown class data type: %s" %class_type)

        sig = np.array(sig)
        assert sig.shape == (dims, t_len/dt)
        class_sig_list.append(sig)

    # write and return
    assert len(class_sig_list) == n_classes

    filename = "./datasets/dataset_%scls_%s_%s_%s_%s" %(class_type, t_len, dims, n_classes, SEED)
    class_desc["class_type"] = class_type
    class_desc["t_len"] = t_len
    class_desc["dims"] = dims
    class_desc["n_classes"] = n_classes
    class_desc["SEED"] = SEED

    np.savez(filename, class_sig_list=class_sig_list, class_desc=class_desc)
    return (class_sig_list, class_desc)

class DataFeed(object):

    def __init__(self, dataset, correct, t_len, filename="derp", log=True):
        self.data_index = 0


        self.time = 0.0
        self.sig_time = 0
        # how often to write the answer into a file
        self.ans_log_period = 0.05
        # how much to pause between questions
        self.pause_time = 0.01
        self.paused = False
        self.q_duration = t_len * dt
        self.correct = correct

        self.qs = dataset
        self.num_items = dataset.shape[0]
        self.dims = dataset.shape[1]
        self.indices = list(np.arange(self.num_items))

        if log:
            self.status = open("results/%s" %filename, "w")
            self.f_r = open("results/%s" %filename, "w")

    def set_answer(self, t, x):
        """just save the answer to a file, like a probe
        saved as [given, correct, paused]

        the expected answer is a one-hot encoded vector, but as long as
        the maximum confidence result is the right answer, it is considered
        correct"""

        if t % self.ans_log_period == 0:
            self.f_r.write("%s, %s, %s" %(x, self.correct[self.indices[self.data_index]], int(self.paused)))


    def feed(self, t):
        """feed the answer into the network

        this is the main state machine of the network"""
        self.time += dt

        if self.time > self.pause_time and self.time > self.q_duration:

            # increment function
            if self.data_index < self.num_items - 1:
                self.data_index += 1
                print("Increment: %s" %self.data_index)
            else:
                print("Shuffling\n")
                self.status.write("Shuffling\n")
                shuffle(self.indices)
                self.data_index = 0

            self.time = 0.0
            self.sig_time = 0
        elif self.time > self.pause_time:
            self.paused = False
            return_val = self.qs[self.indices[self.data_index]][:, self.sig_time]
            self.sig_time += 1
            return return_val
        else:
            self.paused = True
        return np.zeros(self.dims)


def create_feed_net(dataset, correct, t_len):
    """function for feeding data"""
    with nengo.Network(label="feed") as feed:
        feed.d_f = DataFeed(dataset, correct, t_len)
        feed.q_in = nengo.Node(feed.d_f.feed, size_out=dataset.shape[1])
        feed.set_ans = nengo.Node(feed.d_f.set_answer, size_in=dataset.shape[1])


    return feed