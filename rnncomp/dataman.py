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
                   "cont_freq", "cont_amp", "flat"]


def d3_scale(dat, out_range=(-1, 1), in_range=None):
    if in_range is None:
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
            b = 1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))


def ortho_nearest(d):
    p = nengo.dists.UniformHypersphere(surface=True).sample(d, d)
    return np.dot(p, np.linalg.inv(sqrtm(np.dot(p.T, p))))


def mk_cls_dataset(t_len=1, dims=1, n_classes=2, freq=10, class_type="cont_spec", save_dir="./datasets"):
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
        # because you might have multiple examples of a signal signal
        ex = []
        for d_i in range(dims):

            if class_type is "cont_spec":
                """classify using the specific white noise signal"""
                ex.append(
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

                ex = interp1d(v_range, vecs, kind="cubic")(t_range)
                break

            elif class_type is "disc_spec":
                """classify using a specific discrete white-noise
                based signal"""

                # create the random values to interpolate between
                assert dt < freq
                white_size = int(t_len * freq)
                white_vals = np.random.uniform(low=-1, high=1, size=white_size)

                # do the interpolation
                tot_size = int(t_len / dt)
                step_size = int(1.0 / freq / dt)
                n_shape = (white_size, step_size)
                white_noise = (white_vals[:, None] * np.ones(n_shape)).reshape(tot_size)

                assert white_vals.shape[0] < white_noise.shape[0]

                ex.append(white_noise)

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

            elif class_type is "flat":
                """fool-proof flat signals for testing"""
                '''
                flat_range = np.concatenate((
                    np.linspace(-1, -0.25, np.floor(n_classes/2.0)),
                    np.linspace(1, 0.25, np.ceil(n_classes/2.0))
                ))
                '''
                flat_range = np.linspace(0.25, 1, n_classes)
                ex = np.ones((dims, int(t_len/dt)))*flat_range[n_i]
                break

            else:
                raise TypeError("Unknown class data type: %s" % class_type)

        sig.append(ex)
        sig = np.array(sig)
        if sig.shape != (1, dims, t_len/dt):
            print(class_type)
            ipdb.set_trace()

        assert sig.shape == (1, dims, t_len/dt)
        class_sig_list.append(sig)

    # write and return
    assert len(class_sig_list) == n_classes

    class_desc["class_type"] = class_type
    class_desc["t_len"] = t_len
    class_desc["dims"] = dims
    class_desc["n_classes"] = n_classes
    class_desc["SEED"] = SEED

    if save_dir is not None:
        filename = "%s/dataset_%scls_%s_%s_%s_%s" % (save_dir, class_type, t_len, dims, n_classes, SEED)
        np.savez(filename, class_sig_list=class_sig_list, class_desc=class_desc)

    return np.array(class_sig_list), class_desc


def make_correct(dataset, n_classes):
    """make a giant array of correct answers"""
    correct = []
    cls_num = dataset.shape[0]
    sig_num = dataset.shape[1]

    for c_i in range(cls_num):
        cor = np.zeros((sig_num, n_classes), dtype=np.int8)
        cor[:, c_i] = 1
        correct.extend(cor)
    correct = np.array(correct, dtype=np.int8)

    assert correct.shape == (cls_num*sig_num, n_classes)

    return correct


def load_dat_file(fi):
    """stupid hack to let `make_run_args` accept a file
    or just a dat structure

    output format = [signal_class, examples, dimensions, time_steps]"""

    if type(fi) == np.lib.npyio.NpzFile:
        dat = fi["class_sig_list"]
    else:
        dat = fi

    cls_num = dat.shape[0]
    sig_num = dat.shape[1]
    dims = dat.shape[2]
    t_steps = dat.shape[3]

    return dat, cls_num, sig_num, dims, t_steps


def make_run_args_nengo(fi):
    """stop organising by class and get the correct answer
    while including pauses to let the networks reset

    output format = [signal_index, dimensions, time_steps]
    output_format for cor = [signal_index, n_classes]"""

    dat, cls_num, sig_num, dims, t_steps = load_dat_file(fi)

    pause_size = int(PAUSE/dt)

    # append zeros to the questions for pauses
    tot_sigs = int(cls_num*sig_num)
    zer = np.zeros((tot_sigs, dims, pause_size))
    re_zer = dat.reshape((tot_sigs, dims, t_steps))
    final_dat = np.concatenate((zer, re_zer), axis=2)

    zer_shape = (cls_num, dims, pause_size)
    assert np.all(np.zeros(zer_shape) == final_dat[:, :dims, :pause_size])

    # get the correct answer
    cor = make_correct(dat, cls_num)

    return final_dat, cor


def make_run_args_ann(n_dat, n_cor):
    """change Nengo inputs to the Lasagne input

    output format for dat = [time_steps, dimensions]
    output format for cor = [time_steps, 1, n_classes]"""

    dims = n_dat.shape[1]
    t_with_pause = n_dat.shape[2]

    dim_last = n_dat.reshape((-1, t_with_pause, dims))
    final_dat = dim_last.reshape((-1, 1, dims))

    pause_size = int(PAUSE/dt)
    n_classes = n_cor.shape[0]
    tot_sigs = n_cor.shape[1]
    t_steps = t_with_pause - pause_size

    zer = np.zeros((tot_sigs, n_classes, pause_size), dtype=np.int8)
    re_zer = np.repeat(n_cor, t_steps, axis=1).reshape((tot_sigs, n_classes, t_steps))
    cor_with_pause = np.concatenate((zer, re_zer), axis=2)
    # put the time-steps last, with dims first and then transpose
    cor = cor_with_pause.reshape((n_classes, 1, -1)).T
    return final_dat, cor


class DataFeed(object):

    def __init__(self, dataset, correct, t_len, dims, n_classes, filename="derp", log=False):
        self.data_index = 0

        self.time = 0.0
        self.sig_time = 0
        # how often to write the answer into a file
        self.ans_log_period = 0.05
        # how much to pause between questions
        self.pause_time = PAUSE
        self.paused = False
        self.q_duration = t_len
        self.correct = correct

        self.qs = dataset
        self.num_items = dataset.shape[0]
        self.dims = dims
        self.n_classes = n_classes
        self.indices = list(np.arange(self.num_items))
        self.log = log

        if log:
            self.status = open("results/%s" % filename, "w")
            self.f_r = open("results/%s" % filename, "w")

    def close_files(self):
        self.status.close()
        self.f_r.close()

    def set_answer(self, t, x):
        """just save the answer to a file, like a probe
        saved as [given, correct, paused]

        the expected answer is a one-hot encoded vector, but as long as
        the maximum confidence result is the right answer, it is considered
        correct"""

        if t % self.ans_log_period == 0 and self.log:
            self.f_r.write("%s, %s, %s" %(x, self.correct[self.indices[self.data_index]], int(self.paused)))

    def get_answer(self, t):
        """signal for correct answer, maybe should be added as a dimension to feed?"""
        if self.pause_time < self.time < self.q_duration:
            return self.correct[self.indices[self.data_index]]
        else:
            return np.zeros(self.n_classes)

    def feed(self, t):
        """feed the answer into the network

        this is the main state machine of the network"""
        self.time += dt

        if self.time > self.pause_time and self.sig_time >= self.q_duration/dt:

            # increment function
            if self.data_index < self.num_items - 1:
                self.data_index += 1
                #print("Increment: %s" %self.data_index)
            else:
                #print("Shuffling\n")
                if self.log:
                    self.status.write("Shuffling\n")
                shuffle(self.indices)
                self.data_index = 0

            self.time = 0.0
            self.sig_time = 0

        elif self.time > self.pause_time:
            self.paused = False

            q_num = int(self.sig_time - self.pause_time/dt)
            return_val = self.qs[self.indices[self.data_index]][:, q_num]
            self.sig_time += 1
            return return_val

        else:
            #print("Pased")
            self.paused = True

        return np.zeros(self.dims)


def create_feed_net(dataset, correct, t_len, dims, n_classes):
    """create network for feeding data"""
    with nengo.Network(label="feed") as feed:
        feed.d_f = DataFeed(dataset, correct, t_len, dims, n_classes)
        feed.q_in = nengo.Node(feed.d_f.feed, size_out=dims)
        feed.set_ans = nengo.Node(feed.d_f.set_answer, size_in=n_classes)
        feed.get_ans = nengo.Node(feed.d_f.get_answer, size_out=n_classes)

    return feed
