import nengo
import np
from nengo.processes import WhiteSignal

def mk_cls_dataset(t_len, dims, n_classes=2, cont=True):
    """given length t_len, dimensions dim, make number of classes given n_classes in terms of a continuous whitenoise signal"""

    assert n_classes >= 2
    assert dims >= 1
    assert t_len > 0

    class_dict = []
    for n_i in range(n_classes):
        sig = []
        for d_i in range(dims):
            freq = 10
            if cont:
                sig.append(WhiteSignal(t_len, 10, seed=d_i).run(t_len))
            else:
                # grab that sig code from the assignment

        class_dict.append(sig)

    # write and return
    np.savez("./datasets/dataset_%scls_%s_%s_%s" %("cont" if cont else "disc", t_len, dims, n_classes), class_dict=class_dict)
    return class_dict


def feed_data(dataset, t_len):
    """function for feeding data"""

    d_i = 0

    def f(t):
        """feed the data randomly, last dim is correct answer"""

        # shuffle every t_len
        return [dataset[d_i], d_i]

    return f