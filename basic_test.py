# in a nengo network, make a node that just looks up the right answer
import random
import numpy as np
from constants import *

np.random.seed(SEED)
random.seed(SEED)

import nengo
from dataman import *


model = nengo.Network()
t_len = 1000
t_range = np.arange(t_len)
dataset = np.array([np.sin(t_range), -np.sin(t_range), np.cos(t_range)])

with model:
    feed_net = create_feed_net(dataset, t_len)