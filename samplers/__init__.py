from utils import import_submodules

_registered_single_sampler = {}
_registered_multi_sampler = {}


def register_multi_sampler(name):
    """
    A decorator with a parameter.
    This decorator returns a function which the class is passed.
    """
    name = name.lower()
    def _register(sampler):
        if name in _registered_multi_sampler:
            raise ValueError("Name {} already chosen, choose a different name.".format(name))
        _registered_multi_sampler[name] = sampler
        return sampler
    return _register


def register_single_sampler(name):
    """
    A decorator with a parameter.
    This decorator returns a function which the class is passed.
    """
    name = name.lower()

    def _register(sampler):
        if name in _registered_single_sampler:
            raise ValueError("Name {} already chosen, choose a different name.".format(name))
        _registered_single_sampler[name] = sampler
        return sampler
    return _register


def load_multi_sampler_class(name):
    lower_name = name.lower()
    if lower_name not in _registered_multi_sampler:
        raise ValueError("Model {} not found: {}".format(name, list(_registered_multi_sampler.keys())))
    return _registered_multi_sampler[lower_name]


def load_single_sampler_class(name):
    lower_name = name.lower()
    if lower_name not in _registered_single_sampler:
        raise ValueError("Model {} not found: {}".format(name, list(_registered_single_sampler.keys())))
    return _registered_single_sampler[lower_name]


def get_all_single_samplers():
    return list(_registered_single_sampler.keys())


def get_all_multi_samplers():
    return list(_registered_multi_sampler.keys())


def get_all_samplers():
    return get_all_single_samplers() + get_all_multi_samplers()


import_submodules('samplers')
