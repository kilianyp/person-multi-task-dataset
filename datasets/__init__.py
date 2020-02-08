from utils import import_submodules
_registered_datasets = {}

def register_dataset(name):
    """
    A decorator with a parameter.
    This decorator returns a function which the class is passed.
    """
    name = name.lower()

    def _register(dataset):
        if name in _registered_datasets:
            # raising a warning breaks autoreload in jupyer nb
            print("Name {} already chosen, dataset will be overwritten.".format(name))
        _registered_datasets[name] = dataset
        return dataset
    return _register


def load_dataset_class(name):
    lower_name = name.lower()
    if lower_name not in _registered_datasets:
        raise ValueError("Dataset {} not found: {}".format(name, list(_registered_datasets.keys())))
    return _registered_datasets[lower_name]


def get_all_datasets():
    return list(_registered_datasets.keys())

# TODO is importing all modules necessary?
# would be nice to only import the module that is used
# which would be better for sacred
import_submodules('datasets')
