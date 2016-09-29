import numpy as np
import matplotlib.pyplot as plt
import ipdb

from constants import PAUSE


def get_res_info(info_func, info_res, ans, ground, t_len=0.5, sample_every=0.001):
    """return the amount of correct answers and the margin based off the average
    over the whole sample

    Note: this might be hardcoded for 1 dim"""

    ans_len = t_len / sample_every
    assert sample_every < PAUSE
    pause_len = PAUSE / sample_every

    # the total number of question signals to process
    sig_num = int(ans.shape[0] / (ans_len + pause_len))

    # start from 1, because first pause is skipped
    for s_i in xrange(1, sig_num+1):
        # get the time-frame for the q&a
        a_i = s_i - 1
        start = int(a_i * ans_len + s_i * pause_len)
        end = int(s_i * (pause_len + ans_len))

        # get attempted answers
        tmp_ans = ans[start:end]
        sum_ans = np.sum(tmp_ans, axis=0)
        max_idx = sum_ans.argsort()[-2:][::-1]

        # get the maximum at the first time step of the ground signal
        # which holds true for the rest of the signal
        # to get correct answer
        g_i = np.argmax(ground[int(start+ans_len/2)])

        info_func(tmp_ans, max_idx, info_res, g_i)

    return info_res


def get_diff(ans, max_idx, res, grnd_idx):
    """get margin of correctness (mean and std) of the answer

    Note: this might be hardcoded for 1 dim"""

    if max_idx[0] == grnd_idx:
        # if correct, then see how far the second answer was behind
        a_diff = ans[:, grnd_idx] - ans[:, max_idx[1]]
        res['ad_mean'].append(np.mean(a_diff))
        res['ad_std'].append(np.std(a_diff))
    else:
        # if wrong, see how far the correct answer was behind the max
        a_diff = ans[:, grnd_idx] - ans[:, max_idx[0]]
        res['ad_mean'].append(np.mean(a_diff))
        res['ad_std'].append(np.std(a_diff))

    # get distance of correct answer from ideal
    g_diff = ans.shape[0] - np.sum(ans[:, grnd_idx])
    res['gd_mean'].append(np.mean(g_diff))
    res['gd_std'].append(np.std(g_diff))


def get_acc(ans_diff, sig_num):
    return np.where(ans_diff > 0)[0].shape[0] / float(sig_num)


def get_conf(ans, max_idx, res, grnd_idx):
    """get "confusion" over time for each signal

    "confusion" is how distinct the final answer is from the second best answer
    regardless if it is right or not

    Note: this might be hardcoded for 1 dim
    """

    # get how distinct the final answer is from the second best answer
    a_diff = ans[:, max_idx[0]] - ans[:, max_idx[1]]
    res['conf_mean'].append(np.mean(a_diff))
    res['conf_std'].append(np.std(a_diff))
