import numpy as np
import ipdb

from constants import *


# this really needs to be tested better
def get_accuracy(ans, ground, t_len=0.5, sample_every=0.001):
    """return the amount of correct answers and the margin based off the average
    over the whole sample

    assumes sample_every < PAUSE

    Note: this might be hardcoded for 1 dim"""
    t_len = t_len/dt
    pause_len = PAUSE/dt
    # the total number of question signals to process
    sig_num = int(ans.shape[0] / (dt/sample_every) / (t_len+PAUSE/dt))

    ans_diff = np.zeros(sig_num)
    ground_diff = np.zeros(sig_num)

    # TODO: get rid of this for-loop, not urgent because small number of sigs
    # start from 1, because first pause is skipped
    for s_i in xrange(1, sig_num+1):
        # get the time-frame for the q&a
        a_i = s_i - 1
        start = int(a_i*t_len + s_i*pause_len)
        end = int(s_i*(pause_len+t_len))

        # get attempted answers
        tmp_ans = ans[start:end]
        sum_ans = np.sum(tmp_ans, axis=0)
        max_idx = sum_ans.argsort()[-2:][::-1]

        # get the maximum at the first time step of the ground signal
        # which holds true for the rest of the signal
        # to get correct answer
        g_i = np.argmax(ground[start:end][0])

        if max_idx[0] == g_i:
            # if correct, then see how far the second answer was behind
            ans_diff[a_i] = sum_ans[g_i] - sum_ans[max_idx[1]]
        else:
            # if wrong, see how far the correct answer was behind the max
            ans_diff[a_i] = sum_ans[g_i] - sum_ans[max_idx[0]]

        ground_diff[a_i] = t_len - sum_ans[g_i]

    acc = np.where(ans_diff > 0)[0].shape[0] / float(sig_num)

    return acc, ans_diff, ground_diff


def get_conf(ans, ground, t_len=0.5, sample_every=0.001):
    """get "confidence" and "confusion" over time for each signal

    "confusion" is how distinct the final answer is from the second best answer

    "confidence" is how much the final answer deviates

    Note: we can cross-reference this with correct answers later
    """

    t_len = t_len/dt
    pause_len = PAUSE/dt
    # the total number of question signals to process
    sig_num = int(ans.shape[0] / (dt/sample_every) / (t_len+PAUSE/dt))

    confidence = np.zeros((sig_num, t_len))
    confusion = np.zeros((sig_num, t_len))

    # TODO: get rid of this for-loop, not urgent because small number of sigs
    # start from 1, because first pause is skipped
    for s_i in xrange(1, sig_num+1):
        # get the time-frame for the q&a
        a_i = s_i - 1
        start = int(a_i*t_len + s_i*pause_len)
        end = int(s_i*(pause_len+t_len))

        # get attempted and correct answer
        tmp_ans = ans[start:end]
        sum_ans = np.sum(tmp_ans, axis=0)
        max_idx = sum_ans.argsort()[-2:][::-1]

        # get how much the final answer deviates
        confidence[a_i] = tmp_ans[max_idx[0]]

        # get how distinct the final answer is from the second best answer
        confusion[a_i] = tmp_ans[max_idx[0]] - tmp_ans[max_idx[1]]

    return {"confidence": confidence, "confusion": confusion}
