import torch.utils.data
import itertools
from abc import abstractmethod, ABC
import collections
import bisect


class Dataset(torch.utils.data.Dataset, ABC):
    @property
    def header(self):
        return self._header

    @property
    def identifier(self):
        return "{}{}{}".format(self.name, self.filter_fn, self.limit)

    def __init__(self, name, data, header, info, transform=None,
                 loader_fn=None, filter_fn=None, limit=None):
        """
        Args:
            source_file: The path to the source file.
            data_dir: The path where the data is stored relative to the paths
                given in the source file.
            transform: Transformations that are executed on each image.
        """

        self.loader_fn = loader_fn
        self.transform = transform
        self.name = name
        self.limit = limit
        self.filter_fn = filter_fn
        self._header = header
        self.data = data
        self.info = info
        if filter_fn is not None:
            self.data = filter_fn(self.data)
        if limit is not None:
            self.data = self.data[:limit]

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build(cfg):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def before_saving(self, endpoints, data):
        """ Pre-hook before endpoints are written to file """
        return endpoints, data

    def after_saving(self, saved_file_path):
        """ Post-hook after all endpoints are written to file.
            Custom eval scripts can be callled here.
            Can return additional measurements."""
        return dict()


def unique_header(datasets):
    headers = {}
    all_headers = set()
    for d in datasets:
        new_header = set(d.header.keys())
        intersection = all_headers.intersection(new_header)
        if len(intersection):
            for key in intersection:
                old = headers[key]
                new = d.header[key]
                if old != new:
                    print('Duplicate key: {} ({} vs {})'
                                       .format(key, old, new))
        all_headers = all_headers.union(new_header)
        headers.update(d.header)
    return headers


class MultiDataset(torch.utils.data.Dataset):
    """ Only used in combination with a MultiSampler"""
    def __init__(self, datasets):
        if isinstance(datasets, collections.Mapping):
            self.datasets = datasets
        else:
            self.datasets = {}
            for dataset in datasets:
                if dataset.name in self.datasets:
                    raise RuntimeError
                self.datasets[dataset.name] = dataset
    # TODO properly
    @property
    def num_labels(self):
        return sum([dataset.num_labels for dataset in self.datasets.values()])
    @property
    def header(self):
        return unique_header(self.datasets.values())

    @property
    def info(self):
        # TODO if there are two datasets with the same keys,
        # they will overwrite.
        # For pole it would be fine, for num_labels not.
        gathered = {}
        for d in self.datasets.values():
            gathered.update(d.info)
        return gathered

    def __getitem__(self, index):
        # this is called with the indices provided
        # by the sampler
        dataset_index, sample_index = index
        dataset = self.datasets[dataset_index]
        return dataset[sample_index], dataset_index


class ConcatDataset(torch.utils.data.ConcatDataset):
    @property
    def header(self):
        return unique_header(self.datasets)

    @property
    def data(self):
        return itertools.chain.from_iterable([d.data for d in self.datasets])

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx], dataset.name

