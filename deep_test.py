import lasagne
import nengo_lasagne
import nengo
from nengo.processes import PresentInput
import theano
theano.config.floatX = "float32"

import matplotlib.pyplot as plt
import ipdb
import sys

from constants import *


def main(t_len, dims, n_classes, dataset, testset):
    """Test the vanilla RNN with Lasagne"""

    # train up using Lasagne

    N_BATCH = 1
    N_HIDDEN = 10 # does this make the gradient disappear?
    nonlin = lasagne.nonlinearities.tanh
    w_init = lasagne.init.HeUniform

    # accept any one-dimensional vector into the input
    l_in = lasagne.layers.InputLayer(shape=(None,))

    # reshape the input, because the Nengo process outputs a 1 dimensional vector
    # and the layers can't process that
    l_reshape_in = lasagne.layers.ReshapeLayer(l_in, shape=(-1, 1, dims))

    # make the recurrent network
    # Taken from: https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
    l_rec = lasagne.layers.DenseLayer(
        l_reshape_in, num_units=N_HIDDEN,
        nonlinearity=nonlin)

    # output linearity has to be softmax or nothing works
    l_dense = lasagne.layers.DenseLayer(l_rec, num_units=n_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    # train in Nengo

    with nengo.Network() as net, nengo_lasagne.default_config():
        # 'insert_weights' is an optional config setting we can use
        # to control whether a connection weight matrix is inserted
        # for each connection, or whether we just directly connect
        # the output of the pre to the post. in this case we already
        # created all the weight matrices we want in the above
        # Lasagne network.
        net.config[nengo.Connection].set_param("insert_weights",
                                               nengo.params.BoolParam("test", default=False))

        input_node = nengo.Node(output=PresentInput(testset[0], dt))
        
        # insert the convolutional network we defined above
        rnn_layer = nengo_lasagne.layers.LasagneNode(output=l_dense, size_in=dims)

        # output node
        output_node = nengo.Node(size_in=n_classes)

        nengo.Connection(input_node, rnn_layer)
        nengo.Connection(rnn_layer, output_node)

        p = nengo.Probe(output_node)

    sim = nengo_lasagne.Simulator(net)

    sim.train({input_node: dataset[0]}, {output_node: dataset[1]},
              #n_epochs=50, minibatch_size=510,
              optimizer=lasagne.updates.adagrad,
              optimizer_kwargs={"learning_rate": 0.1},
              # since we're doing categorisation, this objective function is fine
              objective=lasagne.objectives.categorical_crossentropy)

    # test the network
    sim.run_steps(testset[0].shape[0])

    output = sim.data[p].squeeze()
    plt.plot(output)
    plt.show()
    ipdb.set_trace()
