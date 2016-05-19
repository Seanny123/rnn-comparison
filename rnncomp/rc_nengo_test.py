# based off https://github.com/tcstewar/testing_notebooks/blob/master/Reservoir.ipynb

import nengo
import numpy as np

import ipdb
import matplotlib.pyplot as plt

from dataman import *
from post import *

def reservoir(t_len, dims, n_classes, alif=False):
    n_neurons = 200
    tau = 0.1

    def train(dataset):

        # this makes sure the reccurent weights don't cause the firing rates to explode
        weights = np.random.uniform(-0.5, 0.5, size=(n_neurons, n_neurons))
        # squaring the weights just increases the gain
        weights *= 1.0 / np.max(np.abs(np.linalg.eigvals(weights))**2)

        # make a model and run it to get spiking data
        train_model = nengo.Network()
        if alif:
            train_model.config[nengo.Ensemble].neuron_type = nengo.AdaptiveLIF()
        with train_model:
            feed_net = create_feed_net(dataset[0], dataset[1], t_len, dims, n_classes) 
            normal = nengo.Node(size_in=dims, size_out=dims)

            state = nengo.Ensemble(n_neurons=n_neurons, dimensions=dims,
                                   radius=np.sqrt(dims), seed=SEED)
            nengo.Connection(state.neurons, state.neurons,
                     transform=weights / n_neurons, synapse=tau, seed=SEED)

            nengo.Connection(feed_net.q_in, state, synapse=None)
            nengo.Connection(state, normal)

            p_sig = nengo.Probe(feed_net.q_in, synapse=None)
            p_target = nengo.Probe(feed_net.get_ans, synapse=None)
            p_spikes = nengo.Probe(state.neurons, synapse=tau)
            p_normal = nengo.Probe(normal, synapse=tau)

        print("training simulation start")
        sim_train = nengo.Simulator(train_model)
        with sim_train:
            sim_train.run((t_len + PAUSE)*dataset[0].shape[0])
        print("training simulation done")

        # pass the spiking data and the target to a solver to get decoding weigths
        solver = nengo.solvers.LstsqL2(reg=0.02)
        decoders, _ = solver(sim_train.data[p_spikes], sim_train.data[p_target])
        return (decoders, weights)

    def test(decoders, weights, testset):
        """run the test data with new decoding weights
        set the decoding weights as transforms on a connection"""

        test_model = nengo.Network()
        if alif:
            test_model.config[nengo.Ensemble].neuron_type = nengo.AdaptiveLIF()
        with test_model:
            feed_net = create_feed_net(testset[0], testset[1], t_len, dims, n_classes)
            output = nengo.Node(size_in=n_classes, size_out=n_classes)
            normal = nengo.Node(size_in=dims, size_out=dims)

            state = nengo.Ensemble(n_neurons=n_neurons, dimensions=dims,
                                   radius=np.sqrt(dims), seed=SEED)
            nengo.Connection(state.neurons, state.neurons,
                             transform=weights / n_neurons, synapse=tau, seed=SEED)

            nengo.Connection(feed_net.q_in, state, synapse=None)
            nengo.Connection(state.neurons, output, transform=decoders.T)
            nengo.Connection(output, feed_net.set_ans, synapse=None)
            nengo.Connection(state, normal)

            p_out = nengo.Probe(output)
            p_correct = nengo.Probe(feed_net.get_ans)
            p_test_sig = nengo.Probe(feed_net.q_in, synapse=None)

        print("test simulation start")
        sim_test = nengo.Simulator(test_model)
        with sim_test:
            sim_test.run((t_len + PAUSE)*testset[0].shape[0])
        print("test simulation done")

        return (sim_test.data[p_out], sim_test.data[p_correct])

    return (train, test)
