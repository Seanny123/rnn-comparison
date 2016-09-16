# in a nengo network, make a node that just looks up the right answer
import random
import numpy as np
from constants import *

np.random.seed(SEED)
random.seed(SEED)

import nengo
from dataman import *
import ipdb

model = nengo.Network()
t_len = 1000
pi_range = np.linspace(-np.pi, np.pi, t_len)
t_range = np.linspace(-np.pi, np.pi, t_len)
simple_x = lambda x: 0.5*x
dataset = np.array([[np.sin(pi_range)], [simple_x(t_range)], [np.tanh(pi_range)]])

#ipdb.set_trace()

with model:
    ans = nengo.Node([0])
    tmp = nengo.Node(size_in=1)

    feed_net = create_feed_net(dataset, np.arange(3), t_len)

    nengo.Connection(ans, feed_net.set_ans)
    nengo.Connection(feed_net.q_in, tmp)
