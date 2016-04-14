import lasagne
import nengo_lasagne
import nengo
from nengo.processes import PresentInput

import ipdb

from constants import *

# REMEMBER: pre-process the datasets to include the pause vefore passing them in
def main(t_len, dims, n_classes, dataset, testset):
    """Test the vanilla RNN with Lasagne"""

    # if not discrete, discretize input signal?
    # train up using Lasagne

    # make number of batches equal to one sig or multiple sigs?
    N_BATCH = 1
    GRAD_CLIP = 100
    N_HIDDEN = 100
    nonlin = lasagne.nonlinearities.tanh
    w_init = lasagne.init.HeUniform

    # `None` indicated a variable batch size
    l_in = lasagne.layers.InputLayer(shape=(None,))

    l_reshape = lasagne.layers.ReshapeLayer(l_in, shape=(-1, dims))

    l_rec = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=w_init(),
        W_hid_to_hid=w_init(),
        nonlinearity=nonlin, only_return_final=True)

    # this is definitely wrong, 
    l_out = lasagne.layers.DenseLayer(l_rec, num_units=n_classes, nonlinearity=nonlin)

    # train in Nengo

    with nengo.Network() as net, nengo_lasagne.default_config():
        # 'insert_weights' is an optional config setting we can use
        # to control whether a connection weight matrix is inserted
        # for each connection, or whether we just directly connect
        # the output of the pre to the post. in this case we already
        # created all the weight matrices we want in the above
        # Lasagne network.
        net.config[nengo.Connection].set_param("insert_weights",
                                               nengo.params.BoolParam(False))

        # is this the problem?
        # input node will just present one input image per timestep
        input_node = nengo.Node(output=PresentInput(testset[0], dt))
        
        # insert the convolutional network we defined above
        rnn_layer = nengo_lasagne.layers.LasagneNode(output=l_out, size_in=dims)

        # output node
        output_node = nengo.Node(size_in=n_classes)

        nengo.Connection(input_node, rnn_layer)
        nengo.Connection(rnn_layer, output_node)

        p = nengo.Probe(output_node)

    sim = nengo_lasagne.Simulator(net)

    # TODO: play with the learning rate to see if you can get it work on little training data
    sim.train({input_node: dataset[0]}, {output_node: dataset[1]},
              n_epochs=1, minibatch_size=N_BATCH,
              optimizer=lasagne.updates.adagrad,
              optimizer_kwargs={"learning_rate": 0.01},
              # since we're doing categorisation, this objective function is fine
              objective=lasagne.objectives.categorical_crossentropy)

    # test the network
    sim.run_steps(testset[0].shape[0])

    output = sim.data[p].squeeze()
    ipdb.set_trace()
