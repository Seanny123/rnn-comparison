from rnncomp.dataman import *
from rnncomp.constants import *

def test_make_correct(dat):
    cor = make_correct(dat, 3)
    assert np.all(cor == np.eye(3))

def test_mk_run_args_nengo(dat):
    assert 1 == 1

def test_mk_run_args_ann(dat):
    assert 1 == 1

def test_mk_cls_dataset_flat():
    t_len = 0.1
    dims = 2
    n_classes = 3

    mk_res = mk_cls_dataset(t_len=t_len, dims=dims, n_classes=n_classes, freq=0,
                            class_type="flat", save_res=False)

    assert dat.shape[0] = n_classes

    for m_i, mag in enumerate(np.linspace(0.25, 1, n_classes)):
        assert np.all(mk_res[m_i][0] == mag * np.ones(dims, int(t_len/dt)))

    assert mk_res[1] == {'SEED': 0, 'class_type': 'flat', 'dims': 2, 'n_classes': 3, 't_len': 0.1}