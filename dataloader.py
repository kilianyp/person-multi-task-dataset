import re
import numpy as np
import torch
import collections
from torch.utils.data.dataloader import _use_shared_memory, numpy_type_map, int_classes, string_classes

def default_collate(batch, keys):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return np.asarray(batch)
    elif isinstance(batch[0], collections.Mapping):
        # adapted instead of using the keys of the first batch elements.
        # create values for all keys, so they can later be indexed
        dic = {}
        for key, header_item in keys.items():
            new_batch = []
            for d in batch:
                # always setting default value also for other datasets
                value = d.get(key, header_item.default)
                new_batch.append(value)
            dic[key] = default_collate(new_batch, keys)
        return dic
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples, keys) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def sanity_check(split_info, batch):
    all_idxs = set()
    for name, idxs in split_info.items():
        all_idxs = all_idxs.union(idxs)

    if len(all_idxs) == len(batch):
        return
    elif len(all_idxs) < len(batch):
        #TODO change that not full batch is calculated from start
        raise RuntimeError("Not full batch is used. {} of {}.".format(len(all_idxs), len(batch)))
    else:
        raise RuntimeError("More idxs are used than there are elements in the batch.")


def count_split(batch_info):
    split = {}
    for idx, name in enumerate(batch_info):
        if not name in split:
            split[name] = [idx]
        else:
            split[name].append(idx)
    return split


def _collate_fn(batch, keys, sanity_check):
    """
    Collate fn for multi-dataset training.
    Each batch is annotated with which idx belongs to
    which dataset.

    Args:
        batch: A list of tuples with (sample, dataset_name), is passed by
               getitem of MultiDataset
    """
    collated_batch = default_collate(batch, keys)
    data, batch_info = collated_batch
    split_info = count_split(batch_info)
    if sanity_check:
        sanity_check(split_info, batch)
    #batch.pop('dataset', None)
    data['split_info'] = split_info
    return data


