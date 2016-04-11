# based off https://github.com/tcstewar/testing_notebooks/blob/master/Reservoir.ipynb

import nengo
import scipy.io

import ipdb
import matplotlib.pyplot as plt

from dataman import *
from post import *

def main(t_len, dims, n_classes, dataset, testset):

    sim_len = dataset[0][0].shape[1]
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

    # pass the spiking data and the target to a solver to get decoding weigths
    # save the decoding weigths?
    solver = nengo.solvers.LstsqL2(reg=0.02)
    # WTF? WHY ARE ALL MY DECODERS ZERO?
    print("getting decoders")
    decoders, info = solver(sim_train.data[p_spikes], sim_train.data[p_target])
    print("done")

    # run the test data with new decoding weights
    # set the decoding weights as transforms on a connection

    test_model = nengo.Network()
    with test_model:
        feed_net = create_feed_net(testset[0], testset[1], t_len, dims)
        output = nengo.Node(size_in=2, size_out=2)

        state = nengo.Ensemble(n_neurons=n_neurons, dimensions=dims)
        weights = np.random.uniform(-0.5, 0.5, size=(n_neurons, n_neurons))
        scale = 1.0 / np.max(np.abs(np.linalg.eigvals(weights)**2))
        weights *= scale
        nengo.Connection(state.neurons, state.neurons, transform=weights / 50, synapse=tau)

        nengo.Connection(feed_net.q_in, state, synapse=None)
        #ipdb.set_trace()
        nengo.Connection(state.neurons, output, transform=decoders.T)
        nengo.Connection(output, feed_net.set_ans, synapse=None)

        p_out = nengo.Probe(output)
        p_correct = nengo.Probe(feed_net.get_ans)

    print("test simulation start")
    sim_test = nengo.Simulator(test_model)
    with sim_test:
        # TODO: get simulation length somehow, from dataset?
        sim_test.run((t_len + PAUSE)*testset[0].shape[0])
    print("test simulation start")

    # TODO: analyse the dataset
    #return get_accuracy(sim.data[p_out], sim.data[p_correct])
    # For now, just plot the results
    plt.plot(sim_test.trange(), sim_test.data[p_out], alpha=0.6)
    plt.plot(sim_test.trange(), sim_test.data[p_correct], alpha=0.6)
    plt.show()
    ipdb.set_trace()
