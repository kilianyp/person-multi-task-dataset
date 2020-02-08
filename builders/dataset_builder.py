from datasets.dataset import Dataset, ConcatDataset
from datasets.reid_dataset import ReidDataset, ConcatReidDataset
from datasets.dummy import DummyReidDataset
from datasets import load_dataset_class, get_all_datasets
from datasets.coco import *
from datasets.utils import *
import os
from utils import format_string_list

def build(cfg):
    if not isinstance(cfg, list):
        cfg = [cfg]
    if len(cfg) > 1:
        datasets = []
        for c in cfg:
            d = build_dataset(c)
            datasets.append(d)
        dataset = build_concat_dataset(datasets)
    else:
        dataset = build_dataset(cfg[0])

    return dataset


def build_loader_fn(name):
    if name == "pil":
        loader_fn = pil_loader
    elif name == "default":
        loader_fn = cv2_loader
    elif name == "cv2":
        loader_fn = cv2_loader
    elif name == "pil2cv":
        loader_fn = pil_2_cv_loader
    elif name == "cv2pil":
        loader_fn = cv_2_pil_loader
    else:
        loaders = ["pil", "cv2", "pil2cv", "cv2pil"]
        raise ValueError("Unknown loader: {}. Choose from {}".format(name, format_string_list(loaders)))

    return loader_fn


def build_filter_fn(name):
    if name is None:
        return None
    elif name == 'filter_junk':
        return filter_junk
    raise ValueError


def build_dataset(cfg):
    name = cfg['name'].lower()
    if name not in get_all_datasets():
        raise ValueError('{} not it Datasets: {}'.format(name, ' ,'.join(get_all_datasets())))
    cls = load_dataset_class(name)

    loader_fn = build_loader_fn(cfg.get('loader_fn', 'default'))
    filter_fn = build_filter_fn(cfg.get('filter_fn'))
    limit = cfg.get('limit')

    return cls.build(cfg, loader_fn=loader_fn, filter_fn=filter_fn, limit=limit)


def build_concat_dataset(datasets):
    reid = False
    for d in datasets:
        if isinstance(d, (ReidDataset, DummyReidDataset)):
            reid = True
        elif isinstance(d, Dataset) and reid:
            raise RuntimeError('Cannot concatenate two different dataset types.')
        elif isinstance(d, Dataset):
            continue
        else:
            raise ValueError("Cannot handle this dataset {}".format(type(d)))

    if reid:
        return ConcatReidDataset(datasets)
    else:
        return ConcatDataset(datasets)

