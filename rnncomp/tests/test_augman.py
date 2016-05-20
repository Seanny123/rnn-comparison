import pytest

from rnncomp.augman import *
from rnncomp.dataman import *

# TODO: should probably not hardcode class numbers...
# TODO: how do you pass arguments to fixtures

@pytest.fixture
def cor(dat):
    """load the dataset answers"""
    return make_correct(dat, 3)

@pytest.fixture
def rng():
    return np.random.RandomState(0)


def test_dat_shuffle(dat, cor, rng):
    shuf_res = dat_shuffle(dat, cor, rng)
    assert fuck == it

def test_ann_repeat(dat, cor, rng)
    ann_repeat(dat, cor, 0.1, rng)