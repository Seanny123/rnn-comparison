import numpy as np
from nengo.processes import WhiteSignal
import matplotlib.pyplot as plt
import nengo

from augman import *
from constants import *

""" Run each augmentation method on both a flat line and a whitenoise signal
and plot the result"""

t_len = 0.5

line_sig = np.ones(t_len/dt)*0.5
line_class_desc = {"t_len":t_len, "dims":1, "n_classes":1}
white_sig = WhiteSignal(t_len, 10, seed=1337).run(t_len)[:, 0]
white_class_desc = {"t_len":t_len, "dims":1, "n_classes":1}

'''
# this just give a ridiculous signal, which might be useful for online learning
# and for seeing if two dissimilar inputs mapped to the same class causes
# problems. Otherwise, it's basically useless.
plt.title("Conv WhiteNoise")
plt.plot(line_sig)
plt.plot(white_sig)
plt.plot(conv_rand_noise(line_sig, t_len))
plt.plot(conv_rand_noise(white_sig, t_len))
plt.show()
'''

# this performs well
plt.title("Add WhiteNoise")
plt.plot(line_sig)
plt.plot(white_sig)
plt.plot(add_rand_noise(line_sig, t_len))
plt.plot(add_rand_noise(white_sig, t_len))
plt.show()

# this performs well
plt.title("Shot Noise")
plt.plot(line_sig)
plt.plot(white_sig)
plt.plot(shot_noise(line_sig, t_len))
plt.plot(shot_noise(white_sig, t_len))
plt.show()

plt.title("Lowpass filt")
plt.plot(line_sig)
plt.plot(white_sig)
plt.plot(nengo.Lowpass(0.01).filt(line_sig))
plt.plot(nengo.Lowpass(0.01).filt(white_sig))
plt.show()

plt.title("Lag noise")
plt.plot(line_sig)
plt.plot(white_sig)
plt.plot(lag(line_sig, t_len))
plt.plot(lag(white_sig, t_len))
plt.show()
