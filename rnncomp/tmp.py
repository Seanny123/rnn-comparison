from rnncomp.dataman import *
from rnncomp.constants import *

def get_strt_end(m_i, t_len, pause_size):
    m_strt = m_i*t_len + (m_i+1)*pause_size
    m_end  = (m_i+1)*(t_len+pause_size)

    z_strt  = m_i*(t_len+pause_size)
    z_end = z_strt + pause_size

    return (m_strt, m_end, z_strt, z_end)

mk_res = mk_cls_dataset(t_len=0.1, dims=2, n_classes=3, freq=0,
                        class_type="flat", save_res=False)
dat = mk_res[0]

n_classes = 3
dims = 2
t_len = int(0.1/dt)
pause_size = int(PAUSE/dt)

_, post_cor = make_run_args_nengo(dat)

for m_i in xrange(n_classes):
    mag_val = np.zeros((t_len, n_classes,))
    mag_val[:, m_i] = 1

    zer_val = np.zeros((pause_size, n_classes, ))

    m_strt, m_end, z_strt, z_end = get_strt_end(m_i, t_len, pause_size)

    assert np.all( post_cor[m_strt:m_end] == mag_val )
    assert np.all( post_cor[z_strt:z_end] == zer_val )