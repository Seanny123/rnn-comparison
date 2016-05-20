import pytest

@pytest.fixture
def dat():
    """load the dataset"""
    mk_res = mk_cls_dataset(t_len=0.1, dims=2, n_classes=3, freq=0,
                            class_type="flat", save_res=False)
    return mk_res[0]