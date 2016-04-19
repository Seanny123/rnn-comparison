# based off https://github.com/tcstewar/testing_notebooks/blob/master/Reservoir.ipynb

import nengo
import numpy as np

import ipdb
import matplotlib.pyplot as plt

from dataman import *
from post import *

def main(t_len, dims, n_classes, dataset, testset):

    n_neurons = 200
    t_neurons = n_neurons * dims
    tau = 0.1

    # this makes sure the reccurent weights don't cause the firing rates to explode
    weights = np.random.uniform(-0.5, 0.5, size=(t_neurons, t_neurons))
    scale = 1.0 / np.max(np.abs(np.linalg.eigvals(weights)**2))
    weights *= scale

    # make a model and run it to get spiking data
    train_model = nengo.Network()
    with train_model:
        feed_net = create_feed_net(dataset[0], dataset[1], t_len, dims, n_classes) 
        normal = nengo.Node(size_in=dims, size_out=dims)

        state = nengo.networks.EnsembleArray(n_neurons, dims, seed=SEED)
        state.add_neuron_input()
        state.add_neuron_output()
        nengo.Connection(state.neuron_input, state.neuron_output,
                 transform=weights / t_neurons, synapse=tau, seed=SEED)

        nengo.Connection(feed_net.q_in, state.input, synapse=None)
        nengo.Connection(state.output, normal)

        p_sig = nengo.Probe(feed_net.q_in, synapse=None)
        p_target = nengo.Probe(feed_net.get_ans, synapse=None)
        p_spikes = nengo.Probe(state.neuron_output, synapse=tau)
        p_normal = nengo.Probe(normal, synapse=tau)

    print("training simulation start")
    sim_train = nengo.Simulator(train_model)
    with sim_train:
        sim_train.run((t_len + PAUSE)*dataset[0].shape[0])
    print("training simulation done")

    plt.plot(sim_train.trange(), sim_train.data[p_sig], alpha=0.6)
    plt.plot(sim_train.trange(), sim_train.data[p_target], alpha=0.6)
    plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
    plt.ylim(-1.1, 1.1)
    #plt.legend()
    plt.show()

    # pass the spiking data and the target to a solver to get decoding weigths
    # save the decoding weigths?
    solver = nengo.solvers.LstsqL2(reg=0.02)
    print("getting decoders")
    # Try a communication channel
    decoders, info = solver(sim_train.data[p_spikes], sim_train.data[p_target])
    #print("rmse: %s" %info["rmse"])

    # run the test data with new decoding weights
    # set the decoding weights as transforms on a connection

    with train_model:
        feed_net.d_f.__init__(testset[0], testset[1], t_len, dims, n_classes)
        output = nengo.Node(size_in=n_classes, size_out=n_classes)

        nengo.Connection(state.neuron_output, output, transform=decoders.T)
        nengo.Connection(output, feed_net.set_ans, synapse=None)
        nengo.Connection(state.output, normal)

        p_out = nengo.Probe(output)
        p_correct = nengo.Probe(feed_net.get_ans)

    print("test simulation start")
    sim_test = nengo.Simulator(train_model)
    with sim_test:
        sim_test.run((t_len + PAUSE)*testset[0].shape[0])
    print("test simulation start")

    # TODO: analyse the dataset
    #return get_accuracy(sim.data[p_out], sim.data[p_correct])
    # For now, just plot the results
    plt.plot(sim_test.trange(), nengo.Lowpass(0.01).filt(sim_test.data[p_out]), alpha=0.6)
    plt.plot(sim_test.trange(), sim_test.data[p_correct], alpha=0.6)
    #plt.plot(sim_train.trange(), sim_train.data[p_normal], alpha=0.4)
    #plt.legend()
    plt.show()
    ipdb.set_trace()
