# try training a linear representation, but without the initial whitenoise stuff

import nengo
import numpy as np
import matplotlib.pyplot as plt
import ipdb

from constants import *

n_neurons = 50
tau = 0.1
dims = 1

# make a model and run it to get spiking data
train_model = nengo.Network()
with train_model:
    q_in = nengo.Node(lambda t: t-1)

    normal = nengo.Node(size_in=dims, size_out=dims)

    state = nengo.Ensemble(n_neurons=n_neurons, dimensions=dims, seed=SEED)
    nengo.Connection(q_in, state, synapse=None)
    nengo.Connection(state, normal)

    p_sig = nengo.Probe(q_in, synapse=None)
    p_target = nengo.Probe(q_in, synapse=None)
    p_spikes = nengo.Probe(state.neurons, synapse=tau)
    p_normal = nengo.Probe(normal, synapse=tau)

print("training simulation start")
sim_train = nengo.Simulator(train_model)
with sim_train:
    sim_train.run(2)
print("training simulation done")

#plt.plot(sim_train.trange(), sim_train.data[p_sig], alpha=0.6)
#plt.plot(sim_train.trange(), sim_train.data[p_target], alpha=0.6)
#plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
#plt.legend()
#plt.show()

# pass the spiking data and the target to a solver to get decoding weigths
# save the decoding weigths?
solver = nengo.solvers.LstsqL2(reg=0.02)
print("getting decoders")
decoders, info = solver(sim_train.data[p_spikes], sim_train.data[p_target])
print("rmse: %s" %info["rmses"])

# run the test data with new decoding weights
# set the decoding weights as transforms on a connection

test_model = nengo.Network()
with test_model:
    q_in = nengo.Node(lambda t: t-1)
    output = nengo.Node(size_in=dims, size_out=dims)
    normal = nengo.Node(size_in=dims, size_out=dims)
    state = nengo.Ensemble(n_neurons=n_neurons, dimensions=dims, seed=SEED)

    nengo.Connection(q_in, state, synapse=None)
    nengo.Connection(state.neurons, output, transform=decoders.T)
    nengo.Connection(state, normal)

    p_out = nengo.Probe(output)
    p_correct = nengo.Probe(q_in)

print("test simulation start")
sim_test = nengo.Simulator(test_model)
with sim_test:
    sim_test.run(2)
print("test simulation start")

plt.plot(sim_test.trange(), nengo.Lowpass(0.01).filt(sim_test.data[p_out]), alpha=0.6)
plt.plot(sim_test.trange(), sim_test.data[p_correct], alpha=0.6)
plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
#plt.legend()
plt.show()
ipdb.set_trace()