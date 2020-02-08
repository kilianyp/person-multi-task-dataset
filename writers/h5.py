import os
import h5py


class h5Writer(object):
    def __init__(self, output_file, shapes):
        self.shapes = shapes
        self.output_file = output_file
        self.position = {}
        self.datasets = {}


    def initialize(self):
        print(self.shapes)
        start_position = 0
        for key, (shape, dtype) in self.shapes.items():
            self.datasets[key] = [self.fh.create_dataset(key, shape=shape, dtype=dtype),
                                  start_position]

    def write(self, **data_dic):
        for key, data in data_dic.items():
            if key not in self.shapes:
                continue
            dataset, start_idx = self.datasets[key]
            end_idx = start_idx + len(data)
            dataset[start_idx:end_idx] = data
            self.datasets[key][1] = end_idx

    def __enter__(self):
        self.fh = h5py.File(self.output_file)
        self.initialize()
        # not sure if this good style
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.fh.close()
        if (exc_type and exc_value and tb):
            print("deleting {}".format(self.output_file))
            os.remove(self.output_file)

class DynamicH5Writer(object):
    """
    Assumes data comes from torch collate_fn
    """
    DEFAULT_SIZE = 1000

    def __init__(self, output_file):
        self.output_file = output_file
        self.position = {}
        self.datasets = {}
        self.free = {}


    def _add_dataset(self, key, data):
        if isinstance(data, list):
            shape = list()
            # assumes string
            dtype = h5py.special_dtype(vlen=str)
        else:
            # first dim is batch
            shape = data.shape[1:]
            dtype = data.dtype

        print("adding", key, shape)
        start_position = 0
        maxshape = tuple([None] + list(shape))
        shape = tuple([self.DEFAULT_SIZE] + list(shape))

        self.datasets[key] = self.fh.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype)
        self.position[key] = start_position
        self.free[key] = self.DEFAULT_SIZE

    def _increase_dataset(self, key):
        dataset = self.datasets[key]
        shape = list(dataset.shape)
        shape[0] += self.DEFAULT_SIZE
        dataset.resize(shape)
        self.free[key] += self.DEFAULT_SIZE


    def write(self, **data_dic):
        for key, data in data_dic.items():
            if key not in self.datasets:
                self._add_dataset(key, data)
            dataset = self.datasets[key]
            start_idx = self.position[key]
            while len(data) > self.free[key]:
                self._increase_dataset(key)

            end_idx = start_idx + len(data)
            dataset[start_idx:end_idx] = data
            self.position[key] = end_idx
            self.free[key] -= len(data)

    def _shrink(self):
        """Shrink datasets to minimal size."""
        for key, dataset in self.datasets.items():
            shape = list(dataset.shape)
            shape[0] -= self.free[key]
            dataset.resize(shape)
            self.free[key] = 0

    def __enter__(self):
        self.fh = h5py.File(self.output_file)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # TODO reshape to remove unnecessary deleted files
        self._shrink()
        self.fh.close()
        if (exc_type and exc_value and tb):
            print("deleting {}".format(self.output_file))
            os.remove(self.output_file)
