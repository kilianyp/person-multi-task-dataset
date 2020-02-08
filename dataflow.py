import collections
from functools import partial


def return_all(tensor_data, split_info):
    return tensor_data


def get_idxs(split_info, datasets):
    idxs = []
    for d in datasets:
        if d in split_info:
            idxs.extend(split_info[d])
    return idxs


def filter(tensor_data, split_info, targets):
    # some models don't always produce an output, save time
    if tensor_data is None:
        return None
    idxs = get_idxs(split_info, targets)
    # if no data is selected just return None
    if len(idxs) == 0:
        return None

    if isinstance(tensor_data, collections.Sequence):
        new_tensor = [data[idxs] for data in tensor_data]
    elif isinstance(tensor_data, collections.Mapping):
        new_tensor = {}
        for key, data in tensor_data.items():
            try:
                new_tensor[key] = filter(data, split_info, targets)
            except TypeError as e:
                print(key, tensor_data[key])
                raise TypeError("Trying to index key {} of type {}.".format(key, type(tensor_data[key])))

        # for some reason this does not work.
#            new_tensor = {key: data[idxs] for key, data in tensor_data.items()}
    else:
        new_tensor = tensor_data[idxs]
    return new_tensor


class DataFlowConfig(object):
    def __init__(self, targets, output_name):
        if not isinstance(targets, list):
            targets = [targets]
        self.targets = targets

        # if datasets is set to all, just return original value without checking during runtime.
        # TODO also check if all datasets are passed
        if targets[0] == "all":
            self.filter_fn = return_all
        else:
            self.filter_fn = partial(filter, targets=self.targets)

        self.output_name = output_name

    def filter(self, tensor_data, split_info):
        # a dictionary of tensors or just a tensor
        return self.filter_fn(tensor_data, split_info)


class DataFlowController(object):
    def __init__(self, cfgs):
        # TODO check if cfgs are identical
        self.cfgs = cfgs

    def split(self, data, split_info):
        split_data = {}
        for cfg in self.cfgs:
            split_data[cfg.output_name] = cfg.filter(data, split_info)
        return split_data


class DummyFlowController(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def split(self, data, split_info):
        return {cfg.output_name: [] for cfg in self.cfgs}
