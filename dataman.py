import nengo
import numpy as np
from nengo.processes import WhiteSignal
from constants import *
from random import shuffle

class_type_list = ["cont_spec", "disc", "cont_packet", "cont_freq", "cont_amp"]

def ortho_nearest(d):
    from scipy.linalg import sqrtm
    p = nengo.dists.UniformHypersphere(surface=True).sample(d, d)
    return np.dot(p, np.linalg.inv(sqrtm(np.dot(p.T, p))))

def mk_cls_dataset(t_len, dims, n_classes=2, class_type="cont_spec"):
    """given length t_len, dimensions dim, make number of classes given n_classes in terms of a continuous whitenoise signal"""

    assert n_classes >= 2
    assert dims >= 1
    assert t_len > 0

    class_dict = []
    # not used for for '_spec' signals
    class_desc = []
    for n_i in range(n_classes):
        sig = []
        for d_i in range(dims):
            freq = 10
            if class_type is "cont_spec":
                # classify based off the specific white noise signal
                sig.append(WhiteSignal(t_len, 10, seed=d_i).run(t_len))
            elif class_type is "orth_spec":
                # classify based off the specific orthgonal weird signal
                raise NotImplementedError("Nope")
                sig.append(WhiteSignal(t_len, 10, seed=d_i).run(t_len))
            elif class_type is "disc_spec":
                # grab that sig code from the assignment
                raise NotImplementedError("Nope")
            elif class_type is "cont_packet":
                # classify using a mix of FM and AM packets
                raise NotImplementedError("Nope")
            elif class_type is "cont_freq":
                # classify based off the frequency of the white noise signal
                raise NotImplementedError("Nope")
            elif class_type is "cont_amp":
                # classify based off the amplitude of the white noise signal
                raise NotImplementedError("Nope")
            else:
                raise TypeError("Unknown class data type: %s" %class_type)


        class_dict.append(sig)

    # write and return
    np.savez("./datasets/dataset_%scls_%s_%s_%s" %(class_type, t_len, dims, n_classes), class_dict=class_dict, class_desc=class_desc)
    return class_dict

### Different augmentation functions ###

"""
def aug_gauss_noise(dataset):

def aug_shot_noise(dataset):

def aug_filt(dataset):

def aug_offset(dataset):

def aug_make_more(dataset):
    # won't work for '_spec' signals
"""

class DataFeed(object):

    def __init__(self, dataset, t_len, filename="derp", log=True):
        self.data_index = 0


        self.time = 0.0
        self.sig_time = 0.0
        # how often to write the answer into a file
        self.ans_log_period = 0
        # how much to pause between questions
        self.pause_time = 0.01
        self.paused = False
        self.q_duration = t_len

        self.qs = dataset
        self.num_items = dataset.shape[1]
        self.indices = list(np.arange(self.num_items))

        if log:
            self.status = open("results/%s" %filename, "w")
            self.f_r = open("results/%s" %filename, "w")

    def set_answer(self, t, x):
        """just save the answer to a file, like a probe
        saved as [given, correct]

        the expected answer is a one-hot encoded vector, but as long as
        the maximum confidence result is the right answer, it is considered
        correct"""

        if t:
            self.f_r.write(x, correct, int(self.paused))


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
                self.list_index = 0

            self.time = 0.0
        elif self.time > self.pause_time:
            self.paused = False
            return_val = self.qs[self.indices[self.data_index]][self.sig_time]
            self.sig_time += dt
            return return_val
        else:
            self.paused = True


def create_feed_net(dataset, t_len):
    """function for feeding data"""
    with nengo.Network(label="feed") as feed:
        feed.d_f = DataFeed(dataset, t_len)
        feed.q_in = nengo.Node(feed.d_f.feed)
        feed.set_ans = nengo.Node(feed.d_f.set_answer, size_in=dataset.shape[1])


    return feed