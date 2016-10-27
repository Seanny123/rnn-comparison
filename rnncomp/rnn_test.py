from constants import *

import lasagne
import nengo_lasagne
import nengo
from nengo.processes import PresentInput
import numpy as np
import theano
theano.config.floatX = "float32"

import ipdb

class Ident(lasagne.init.Initializer):
    """Initialize weights with constant value.
    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def sample(self, shape):
        assert shape[0] == shape[1]
        return lasagne.utils.floatX(np.eye(shape[0]))


def vanilla(dims, n_classes):

    def train(datset, corset, w_rec_init=lasagne.init.HeUniform, nonlin=lasagne.nonlinearities.tanh, learning_rate=0.01):
        """Test the vanilla RNN with Lasagne"""

        # train up using Lasagne

        N_BATCH = 1
        GRAD_CLIP = 100
        N_HIDDEN = 50  # does this make the gradient disappear?
        w_init = lasagne.init.HeUniform

        # accept any one-dimensional vector into the input
        l_in = lasagne.layers.InputLayer(shape=(None,))

        # reshape the input, because the Nengo process outputs a 1 dimensional vector # THAT'S NOT TRUE
        # and RNNs can't process that
        l_reshape_in = lasagne.layers.ReshapeLayer(l_in, shape=(-1, 1, dims))

        # make the recurrent network
        # Taken from: https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
        l_rec = lasagne.layers.RecurrentLayer(
            l_reshape_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
            W_in_to_hid=w_init(),
            W_hid_to_hid=w_rec_init(),
            nonlinearity=nonlin)

        l_dense = lasagne.layers.DenseLayer(l_rec, num_units=n_classes, nonlinearity=lasagne.nonlinearities.softmax)

        # train in Nengo

        with nengo.Network() as net, nengo_lasagne.default_config():
            # 'insert_weights' is an optional config setting we can use
            # to control whether a connection weight matrix is inserted
            # for each connection
            net.config[nengo.Connection].set_param("insert_weights",
                                                   nengo.params.BoolParam("test", default=False))

            input_node = nengo.Node(output=PresentInput(datset, dt))
            
            # insert the recurrent network we defined above
            rnn_layer = nengo_lasagne.layers.LasagneNode(output=l_dense, size_in=dims)

            # output node
            output_node = nengo.Node(size_in=n_classes)

            nengo.Connection(input_node, rnn_layer)
            nengo.Connection(rnn_layer, output_node)

            # sample_every is really not working on this thing
            p_out = nengo.Probe(output_node, sample_every=sample_every)

        sim = nengo_lasagne.Simulator(net)

        sim.train({input_node: datset}, {output_node: corset},
                  #n_epochs=3, minibatch_size=None,
                  optimizer=lasagne.updates.adagrad,
                  optimizer_kwargs={"learning_rate": learning_rate},
                  # since we're doing categorisation, this objective function is fine
                  objective=lasagne.objectives.categorical_crossentropy)
        return sim, p_out

    def test(sim, testset, p_out):
        # test the network
        sim.run_steps(testset.shape[0])
        sample_step = int(sample_every / dt)
        return sim.data[p_out].squeeze()[::sample_step]

    return train, test
