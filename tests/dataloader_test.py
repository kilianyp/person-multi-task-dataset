from dataloader import sanity_check, default_collate
import numpy as np
from torch.utils.data.dataloader import default_collate
import pytest


def test_sanity_check():
    idxs1 = [0, 1, 2]
    idxs2 = [3, 4, 5]
    split_info = {
            'test1': idxs1,
            }

    sanity_check(split_info, idxs1)


    with pytest.raises(RuntimeError):
        sanity_check(split_info, idxs1 + idxs2)

    split_info['test2'] = idxs2
    with pytest.raises(RuntimeError):
        sanity_check(split_info, idxs1)


def test_default_collate():
    data = {'pid': '01', 'img': np.ndarray((5, 5))}
    batch = [data, data, data]

    keys = {'pid': -1, 'img': 'test'}
    data = ['0', '03']
    batch = default_collate(data)
    print (batch)
