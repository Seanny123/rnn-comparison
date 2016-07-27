import pickle

import lasagne
import matplotlib.pyplot as plt
import nengo # note: requires the `nengo_lasagne_compat` branch currently
import nengo_lasagne
import numpy as np
import theano

from lasagne import nonlinearities as nl
from nengo.processes import PresentInput

theano.config.floatX = "float32"

# load the inputs/targets (download the dataset at
# http://deeplearning.net/data/mnist/mnist.pkl.gz)
with open("mnist.pkl", "rb") as f:
    train, _, test = pickle.load(f)
targets = np.zeros((train[1].shape[0], 1, 10), dtype=np.float32)
targets[np.arange(train[1].shape[0]), :, train[1]] = 1.0

# input layer
l = lasagne.layers.InputLayer(shape=(None,))

# reshape it into the (whatever, n_channels, 28, 28) image shape
l = lasagne.layers.ReshapeLayer(l, shape=(-1, 1, 28, 28))

# 2 convolution/pooling layers
for _ in range(2):
    l = lasagne.layers.Conv2DLayer(l, num_filters=32, filter_size=(5, 5),
                                   nonlinearity=nl.rectify,
                                   W=lasagne.init.HeNormal(gain="relu"))
    l = lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2))

# dense layer
l = lasagne.layers.DenseLayer(l, num_units=256, nonlinearity=nl.rectify,
                              W=lasagne.init.HeNormal(gain='relu'))

# dropout
l = lasagne.layers.DropoutLayer(l, p=0.5)

# output layer
l = lasagne.layers.DenseLayer(l, num_units=10, nonlinearity=nl.softmax)

with nengo.Network() as net, nengo_lasagne.default_config():
    # 'insert_weights' is an optional config setting we can use
    # to control whether a connection weight matrix is inserted
    # for each connection, or whether we just directly connect
    # the output of the pre to the post. in this case we already
    # created all the weight matrices we want in the above
    # Lasagne network.
    net.config[nengo.Connection].set_param("insert_weights",
                                           nengo.params.BoolParam(False))

    # input node will just present one input image per timestep
    input_node = nengo.Node(output=PresentInput(test[0], 0.001))
    
    # insert the convolutional network we defined above
    conv_layers = nengo_lasagne.layers.LasagneNode(output=l, size_in=784)
    
    # output node
    output_node = nengo.Node(size_in=10)

    nengo.Connection(input_node, conv_layers)
    nengo.Connection(conv_layers, output_node)

    p = nengo.Probe(output_node)

sim = nengo_lasagne.Simulator(net)

# we'll use a different optimizer function here, nesterov_momentum.
# we'll also use a different objective, categorical_crossentropy,
# as opposed to the default squared error we were using previously.
sim.train({input_node: train[0][:, None]}, {output_node: targets},
          n_epochs=10, minibatch_size=500,
          optimizer=lasagne.updates.adagrad,
          optimizer_kwargs={"learning_rate": 0.01},
          objective=lasagne.objectives.categorical_crossentropy)