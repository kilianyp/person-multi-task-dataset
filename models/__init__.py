from utils import import_submodules
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

_registered_models = {}
_imported = False


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def infere(self, data):
        with torch.no_grad():
            endpoints = self(data, self.endpoints)
        return endpoints


def register_model(name):
    """
    A decorator with a parameter.
    This decorator returns a function which the class is passed.
    """
    name = name.lower()

    def _register(model):
        if name in _registered_models:
            print("Name {} already chosen, model will be overwritten.".format(name))
        _registered_models[name] = model
        return model
    return _register


def load_model_class(name):
    lower_name = name.lower()
    if lower_name not in _registered_models:
        raise ValueError("Model {} not found: {}".format(name, list(_registered_models.keys())))
    return _registered_models[lower_name]


def get_all_models():
    return list(_registered_models.keys())


# TODO is importing all modules necessary?
# would be nice to only import the module that is used
# which would be better for sacred
# maybe easier to control later throug sacred
import_submodules('models')
