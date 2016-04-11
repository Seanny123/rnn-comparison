# in a nengo network, make a node that just looks up the right answer
import random
import numpy as np
from constants import *

np.random.seed(SEED)
random.seed(SEED)

import nengo
from dataman import *
import ipdb


def make_run_args(fi, dims, t_steps):
    """reshape before passing (stop organising by class) 
    and get the correct-ans and pass that too"""
    dat = fi["class_sig_list"]
    cls_num = dat.shape[0]
    sig_num = dat.shape[1]
    return (dat.reshape((int(cls_num*sig_num), dims, t_steps)), make_correct(dat))

# load a dataset for training
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
desc = fi["class_desc"].item()
dataset = make_run_args(fi, desc["dims"], int(desc["t_len"]/dt))

# do the same for the test set
fi = np.load("datasets/dataset_flatcls_0.5_2_3_0.npz")
testset = make_run_args(fi, desc["dims"], int(desc["t_len"]/dt))

t_len = desc["t_len"]
dims = desc["dims"]
n_classes = desc["n_classes"]

n_neurons = 50
tau = 0.1

# make a model and run it to get spiking data
train_model = nengo.Network()
with train_model:
    feed_net = create_feed_net(dataset[0], dataset[1], t_len, dims)

    state = nengo.Ensemble(n_neurons=n_neurons, dimensions=dims)
    nengo.Connection(feed_net.q_in, state, synapse=None)

    p_sig = nengo.Probe(feed_net.q_in, synapse=None)
    p_target = nengo.Probe(feed_net.get_ans, synapse=None)
    p_spikes = nengo.Probe(state.neurons, synapse=tau)

    weights = np.random.uniform(-0.5, 0.5, size=(n_neurons, n_neurons))
    scale = 1.0 / np.max(np.abs(np.linalg.eigvals(weights)**2))
    weights *= scale
    nengo.Connection(state.neurons, state.neurons, transform=weights / 50, synapse=tau)

print("training simulation start")
sim_train = nengo.Simulator(train_model)
with sim_train:
    sim_train.run((t_len + PAUSE)*dataset[0].shape[0])
print("training simulation done")

plt.plot(sim_train.trange(), sim_train.data[p_sig], alpha=0.6)
plt.plot(sim_train.trange(), sim_train.data[p_target], alpha=0.6)
plt.show()
ipdb.set_trace()