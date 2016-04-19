import numpy as np
import ipdb

from constants import *

def get_accuracy_win(ans, ground, win=100):
    """return the amount of correct, the amount of incorrect answers
    based off the average in and outside the window"""

def get_accuracy(ans, ground, t_len, sample_every=0.001):
    """return the amount of correct answers and the margin based off the average
    over the whole sample

    assumes sample_every < PAUSE"""
    t_len = t_len/dt
    pause_len = PAUSE/dt
    sig_num = ans.shape[0] / (dt/sample_every) / (t_len+PAUSE)

    ans_diff = np.zeros(sig_num)
    ground_diff = np.zeros(sig_num)

    # TODO: get rid of this for-loop, not urgent because small number of sigs
    for s_i in xrange(1, sig_num+1):
        start = int((s_i-1)*t_len + s_i*pause_len)
        end = int(s_i*(pause_len+t_len))

        tmp_ans = ans[start:end]
        g_i = np.argmax(ground[start:end][0])

        sum_ans = np.sum(tmp_ans, axis=0)
        max_idx = sum_ans.argsort()[-2:][::-1]
        if max_idx[0] == g_i:
            # if correct, then see how far the second answer was behind
            ans_diff[s_i] = sum_ans[g_i] - sum_ans[max_idx[1]]
        else:
            # if wrong, see how far the correct answer was behind the max
            ans_diff[s_i] = sum_ans[max_idx[0]] - sum_ans[g_i]

        ground_diff[s_i] = sum_ans[g_i] - t_len

    acc = np.where(ans_diff >= 0)[0] / sig_num

    return (acc, ans_diff, ground_diff)