from samplers.pk_sampler import PKSampler
from samplers.batch_sampler import BatchSampler
from samplers.sequential_sampler import SequentialSampler
from samplers.random_sampler import RandomSampler
from samplers.multi_sampler import RandomSamplerLongest, ConcatenatedSamplerLongest, ConcatenatedSamplerShortest, SwitchingSamplerLongest
from samplers.multi_sampler import RandomSamplerLongestKeep, RandomSamplerShortest, RandomSamplerLengthWeighted
from samplers.multi_sampler import SwitchingSamplerShortest
from samplers.multi_sampler import get_next
from builders.dataset_builder import build_concat_dataset
import builders.sampler_builder as sampler_builder
import builders.dataloader_builder as dataloader_builder
from datasets.dataset import MultiDataset

from datasets.reid_dataset import ConcatReidDataset
import pytest
from samplers.pk_sampler import create_pids2idxs
from datasets.dummy import create_dummy_data, create_dummy_pid_data, DummyDataset

from torch.utils.data import DataLoader
from builders.dataloader_builder import build_collate_fn
import time
import numpy as np




@pytest.mark.parametrize("drop_last", [True, False])
def test_pk_sampler(drop_last):
    # tests drop last
    # 
    # TODO partial

    P = 6
    K = 10
    num_pids = 40
    dummy_cfg = {
        "name": "dummy",
        "num_pids": num_pids,
        "size": 350,
        "data_dir": "/"
    }

    pk_cfg = {
        "type": "pk",
        "dataset": dummy_cfg,
        "P": P,
        "K": K,
        "drop_last": drop_last
    }

    sampler, dataset = sampler_builder.build(pk_cfg)

    pk_counter = {}
    for pid in sampler.pids:
        pk_counter[pid] = 0
    iterations = 0
    for idxs in sampler:
        iterations += 1
        pk_counter_batch = {}
        for idx in idxs:
            data = dataset[idx]
            pid = data['pid']
            pk_counter[pid] += 1
            if pid in pk_counter_batch:
                pk_counter_batch[pid] += 1
            else:
                pk_counter_batch[pid] = 1

        if drop_last:
            assert len(pk_counter_batch) == P

        for key, value in pk_counter_batch.items():
            assert value == K, pk_counter_batch

    assert iterations == len(sampler)
    assert sampler.completed_iter == 1
    # overall tests
    not_used_pids = 0
    for key, value in pk_counter.items():
        if value == 0:
            not_used_pids += 1
            continue
        assert value == K

    if not drop_last:
        assert not_used_pids == 0
    else:
        assert not_used_pids == num_pids % P


@pytest.mark.parametrize("num_workers, sampler",
                         [(1, "random_sampler_longest"),
                          (4, "random_sampler_longest"),
                          (4, "concatenated_longest")])
def test_multi_dataset_loader(num_workers, sampler):
    P = 6
    K = 10
    dummy_cfg1 = {
        "name": "dummy",
        "num_pids": 100,
        "id": "dummy1",
        "size": 500,
        "data_dir": "/"
    }

    pk_cfg = {
        "type": "pk",
        "dataset": dummy_cfg1,
        "P": P,
        "K": K,
        "drop_last": True
    }

    dummy_cfg2 = {
        "name": "dummy",
        "id": "dummy2",
        "size": 750,
        "data_dir": "/"

    }
    sequential_cfg = {
        "type": "sequential",
        "dataset": dummy_cfg2,
        "batch_size": 70
    }


    sampler_cfg = {
        "type": sampler,
        "samplers": {
            "sampler1": pk_cfg,
            "sampler2": sequential_cfg
        }
    }
    dataloader_cfg = {
        "sampler": sampler_cfg,
        "num_workers": num_workers
    }

    dataloader = dataloader_builder.build(dataloader_cfg)


    start = time.time()
    for batch_data in dataloader:
        actual_split = {"dummy1": [], "dummy2": []}
        for idx, dataset_name in enumerate(batch_data['path']):
            if dataset_name == "dummy1":
                actual_split["dummy1"].append(idx)
            elif dataset_name == "dummy2":
                actual_split["dummy2"].append(idx)
            else:
                raise RuntimeError

        sampler_info = batch_data['split_info']
        for dataset, idxs in sampler_info.items():
            assert len(idxs) == len(actual_split[dataset])
            for a, b in zip(idxs, actual_split[dataset]):
                assert a == b

        idxs1 = sampler_info.get("dummy1")
        idxs2 = sampler_info.get("dummy2")

        if idxs1:
            print(np.array(batch_data['pid'])[idxs1])
        if idxs2:
            print(np.array(batch_data['pid'])[idxs2])

    print("Took {}".format(time.time()-start))


def test_pk_concat_dataset():
    P = 4
    K = 3
    num_pids1 = 30
    num_pids2 = 40
    dummy_cfg1 = {
        "name": "dummy_reid",
        "size": 200,
        "num_pids": num_pids1,
        "data_dir": "/"
    }

    dummy_cfg2 = {
        "name": "dummy_reid",
        "size": 300,
        "num_pids": num_pids2,
        "data_dir": "/"
    }

    sampler_cfg = {
        "type": "pk",
        "P": P,
        "K": K,
        "dataset": [dummy_cfg1, dummy_cfg2]
    }


    pk_sampler, dataset = sampler_builder.build(sampler_cfg)
    assert len(pk_sampler.pids) == num_pids1 + num_pids2
    steps = 0
    old_pid = -1
    pid_counter = K
    for batch in pk_sampler:
        steps += 1
        for idx in batch:
            # returns also dataset name
            data, dataset_name = dataset[idx]
            pid = data['pid']
            if old_pid == pid:
                pid_counter += 1
            else:
                assert pid_counter == K
                old_pid = pid
                pid_counter = 1


    assert steps == len(pk_sampler)

def test_get_next():
    size = 100
    class DummyIterator(object):
        def __init__(self, size):
            self.size = size
            self.n = 0

        def __iter__(self):
            self.n = 0
            return self
        
        def __next__(self):
            if self.n < self.size:
                n = self.n
                self.n += 1
                return n
            raise StopIteration

        def reset(self):
            self.n = 0
            
    a = DummyIterator(10)
    for i in range(1, size + 1):
        d = get_next(a)

    assert i == size


def test_combined_pk_pk_sampler():
    size1 = 200
    size2 = 300
    num_pids1 = 30
    num_pids2 = 40
    
    dataset1 = DummyDataset(lambda : create_dummy_pid_data(size1, num_pids1, "d1"), "dummy1")
    dataset2 = DummyDataset(lambda : create_dummy_pid_data(size2, num_pids2, "d2"), "dummy2")

    P = 4
    K = 3
    pk_sampler1 = PKSampler(P, K, dataset1, drop_last=True)
    pk_sampler2 = PKSampler(P, K, dataset2, drop_last=True)

    sampler = ConcatenatedSamplerLongest( [pk_sampler1, pk_sampler2])
    for batch in sampler:
        for idx in batch:
            pass
            #print(dataset[idx])


def test_longest_concatenated_sampler():
    size1 = 70
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)
    
    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)

    sampler = ConcatenatedSamplerLongest( [sampler1, sampler2])
    
    print(len(sampler))
    for idx, batch in enumerate(sampler):
        pass
    
    correct = size1 if size1 > size2 else size2
    assert idx + 1 == correct

def test_shortest_concatenated_sampler():
    size1 = 70
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)
    
    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)

    sampler = ConcatenatedSamplerShortest([sampler1, sampler2])
    
    print(len(sampler))
    for idx, batch in enumerate(sampler):
        pass
    
    correct = size1 if size1 < size2 else size2
    assert idx + 1 == correct


def test_switching_sampler_longest():
    size1 = 70
    size2 = 100
    size3 = 150
    name1 = "Dummy1"
    name2 = "Dummy2"
    name3 = "Dummy3"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)
    dataset3 = DummyDataset(lambda: create_dummy_data(size3, name3), name3)

    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)
    sampler3 = BatchSampler(SequentialSampler(dataset3), 1, True)

    sampler = SwitchingSamplerLongest([sampler1, sampler2, sampler3])

    print(len(sampler))
    for idx, batch in enumerate(sampler):
        if idx < 70 * 3:
            if idx % 3 == 0:
                assert batch[0][0] == name1
            elif idx % 3 == 1:
                assert batch[0][0] == name2
            elif idx % 3 == 2:
                assert batch[0][0] == name3
        elif idx < 70 * 3 + 30 * 2:
            if idx % 2 == 0:
                assert batch[0][0] == name2
            elif idx % 2 == 1:
                assert batch[0][0] == name3
        else:
            assert batch[0][0] == name3
    
    correct = size1 + size2 + size3
    assert idx + 1 == correct


def test_combined_pk_seq_sampler():
    num_pids = 40
    pk_cfg = {
        "P": 6,
        "K": 2
    }
    sequential_cfg = {
        "batch_size": 5
    }
    size = 150
    size2 = 200
    dataset1 = DummyDataset(lambda : create_dummy_pid_data(size, num_pids), "dummy1")
    dataset2 = DummyDataset(lambda : create_dummy_pid_data(size2, num_pids), "dummy2")

    pk_sampler = PKSampler.build(dataset1, pk_cfg)
    seq_sampler = SequentialSampler.build(dataset2, sequential_cfg)

    sampler = ConcatenatedSamplerLongest([pk_sampler, seq_sampler])
    batches_seq = len(seq_sampler)
    batches_pk = len(pk_sampler)
    iterations_pk = batches_seq // batches_pk
    for batch in sampler:
        batch_counter = {dataset1.name: 0, dataset2.name: 0}
        for dataset, idx in batch:
            batch_counter[dataset] += 1
        assert batch_counter[dataset1.name] == pk_sampler.batch_size
        assert batch_counter[dataset2.name] == seq_sampler.batch_size

    print(iterations_pk, pk_sampler.completed_iter)
    assert pk_sampler.completed_iter == iterations_pk


def test_create_pid2idxs():
    num_pids = 40
    dataset = DummyDataset(lambda: create_dummy_pid_data(200, num_pids), "dummy")

    pid2idxs = create_pids2idxs(dataset)
    assert len(pid2idxs) == num_pids


def test_random_sampler():
    dataset = DummyDataset(lambda: create_dummy_data(100), "dummy")
    sampler = RandomSampler(dataset)
    idxs1 = []
    for idxs in sampler:
        idxs1.append(idxs)

    idxs2 = []
    for idxs in sampler:
        idxs2.append(idxs)

    print(idxs1, idxs2)
    assert idxs1 != idxs2

def test_reset():
    size1 = 10
    size2 = 20
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)

    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)

    sampler = RandomSamplerShortest( [sampler1, sampler2])

    print(len(sampler))
    for _ in range(3):
        for idx, batch in enumerate(sampler):
            print(idx, batch)


def test_random_sampler_longest_keep():
    """
    TODO this is a check not a test.
    """
    size1 = 10
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)

    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)

    sampler = RandomSamplerLongestKeep( [sampler1, sampler2])

    print(len(sampler))
    for idx, batch in enumerate(sampler):
        print(idx, batch)


def test_random_sampler_length_weighted():
    """
    TODO this is a check not a test.
    """
    size1 = 1
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)

    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)

    sampler = RandomSamplerLengthWeighted([sampler1, sampler2], [1, 1])

    print(len(sampler))
    for idx, batch in enumerate(sampler):
        print(idx, batch)


def test_random_sampler_length_equal():
    """
    TODO this is a check not a test.
    """
    size1 = 1
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_data(size1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)

    sampler1 = BatchSampler(SequentialSampler(dataset1), 1, True)
    sampler2 = BatchSampler(SequentialSampler(dataset2), 1, True)
    
    cfg = {"weights": "equal"}
    sampler = RandomSamplerLengthWeighted.build([sampler1, sampler2], cfg)

    print(len(sampler))
    for idx, batch in enumerate(sampler):
        print(idx, batch)
