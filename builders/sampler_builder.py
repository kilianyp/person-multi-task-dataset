from builders import dataset_builder
from datasets.dataset import MultiDataset
from samplers import load_multi_sampler_class, get_all_multi_samplers
from samplers import load_single_sampler_class, get_all_single_samplers
from samplers import get_all_samplers


def clean(name):
    # backwards compability
    if name.endswith('_sampler'):
        return name[:-len('_sampler')]
    return name


def build(cfg):
    type = cfg['type'].lower()
    type = clean(type)
    if type in get_all_multi_samplers():
        return build_multi_sampler(type, cfg)
    elif type in get_all_single_samplers():
        return build_single_sampler(type, cfg)
    else:
        raise ValueError("{} not found. Choose from {}.".format(type, get_all_samplers()))


def build_multi_sampler(type, cfg):
    cfgs = cfg['samplers']
    samplers = []
    datasets = []
    for name, c in cfgs.items():
        s_type = c['type'].lower()
        s_type = clean(s_type)
        sampler, dataset = build_single_sampler(s_type, c)
        samplers.append(sampler)
        datasets.append(dataset)

    multi_dataset = MultiDataset(datasets)
    multi_sampler_class = load_multi_sampler_class(type)
    multi_sampler = multi_sampler_class.build(samplers, cfg)
    return multi_sampler, multi_dataset


def build_single_sampler(type, cfg):
    dataset = dataset_builder.build(cfg['dataset'])

    single_sampler_class = load_single_sampler_class(type)
    sampler = single_sampler_class.build(dataset, cfg)

    return sampler, dataset
