from rnncomp.dataman import *
from rnncomp.constants import *


# TODO: test multiple sizes of of dat? signums? so much more coverage...


def test_make_correct(dat):
    n_classes = 3

    cor = make_correct(dat, n_classes)
    assert np.all(cor == np.eye(n_classes))

def get_strt_end(m_i, t_len, pause_size):
    m_strt = m_i*t_len + (m_i+1)*pause_size
    m_end  = (m_i+1)*(t_len+pause_size)

    z_strt  = m_i*(t_len+pause_size)
    z_end = z_strt + pause_size

    return (m_strt, m_end, z_strt, z_end)

def test_mk_run_args_nengo_cor(dat):
    n_classes = 3
    dims = 2
    t_len = int(0.1/dt)
    pause_size = int(PAUSE/dt)

    _, post_cor = make_run_args_nengo(dat)

    for m_i in xrange(n_classes):
        mag_val = np.zeros((t_len, n_classes,))
        mag_val[:, m_i] = 1

        zer_val = np.zeros((pause_size, n_classes,))

        m_strt, m_end, z_strt, z_end = get_strt_end(m_i, t_len, pause_size)

        assert np.all( post_cor[m_strt:m_end] == mag_val )
        assert np.all( post_cor[z_strt:z_end] == zer_val )

def test_mk_run_args_nengo_dat(dat):
    n_classes = 3
    dims = 2
    t_len = int(0.1/dt)
    pause_size = int(PAUSE/dt)

    post_dat, _ = make_run_args_nengo(dat)

    for m_i, mag in enumerate(np.linspace(0.25, 1, n_classes)):
        mag_val = mag * np.ones((t_len, dims,))

        zer_val = np.zeros((pause_size, dims, ))

        m_strt, m_end, z_strt, z_end = get_strt_end(m_i, t_len, pause_size)

        assert np.all( post_dat[m_strt:m_end] == mag_val )
        assert np.all( post_dat[z_strt:z_end] == zer_val )

def test_mk_run_args_ann(dat):
    n_classes = 3
    dims = 2
    t_len = 0.1
    post_dat, post_cor = make_run_args_ann(dat)

    for m_i, mag in enumerate(np.linspace(0.25, 1, n_classes)):
        assert np.all( post_dat[m_i] == mag * np.ones((dims, int(t_len/dt),)) )

    assert np.all(post_cor == np.eye(n_classes))

def test_mk_cls_dataset_flat():
    t_len = 0.1
    dims = 2
    n_classes = 3

    mk_res = mk_cls_dataset(t_len=t_len, dims=dims, n_classes=n_classes, freq=0,
                            class_type="flat", save_res=False)

    assert mk_res[0].shape[0] == n_classes

    for m_i, mag in enumerate(np.linspace(0.25, 1, n_classes)):
        assert np.all( mk_res[0][m_i] == mag * np.ones((dims, int(t_len/dt),)) )

    assert mk_res[1] == {'SEED': 0, 'class_type': 'flat',
                         'dims': dims, 'n_classes': n_classes, 't_len': t_len}

def test_datafeed(dat):
    """run the datafeed and see if it behaves as expected"""