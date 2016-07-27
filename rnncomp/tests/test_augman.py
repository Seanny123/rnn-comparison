import pytest

from rnncomp.augman import *
from rnncomp.dataman import *

import ipdb

# TODO: should probably not hardcode class numbers...
# TODO: how do you pass arguments to fixtures

@pytest.fixture
def cor(dat):
    """load the dataset answers"""
    return make_correct(dat, 3)


def test_dat_shuffle(dat, cor, rng):
    """basic shuffle test"""

    shuf_dat, shuf_cor = dat_shuffle(dat, cor, rng)
    assert not np.all(shuf_dat == dat)
    assert shuf_dat.shape == dat.shape
    assert not np.all(shuf_cor == cor)
    assert shuf_cor.shape == cor.shape


def test_dat_repeat(dat, cor, rng):
    t_len = 0.1/dt
    rep_dat, rep_cor = dat_repeat(dat, cor, 0.1, rng)

    assert rep_dat[] == rep_dat[]
    assert rep_cor[0] == rep_cor[1]
