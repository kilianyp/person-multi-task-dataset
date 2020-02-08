import sys
import os
path = os.path.abspath('./sacred')
sys.path = [path] + sys.path
import glob
import pytest

import builders.sampler_builder as sampler_builder
from samplers.pk_sampler import PKSampler
from samplers.sequential_sampler import SequentialSampler
from samplers.batch_sampler import BatchSampler
from samplers.multi_sampler import MultiSampler, ConcatenatedSamplerLongest, RandomSamplerLengthWeighted


import builders.transform_builder as transform_builder
from datasets.mpii import Mpii

import builders.loss_builder as loss_builder
from losses.triplet_loss import BatchHard
from losses.multi_loss import MultiLoss, SingleLoss

import builders.dataset_builder as dataset_builder
from datasets.dataset import Dataset, MultiDataset, ConcatDataset
from datasets.reid_dataset import ConcatReidDataset, ReidDataset
from datasets.pose_dataset import PoseDataset

import builders.evaluation_builder as evaluation_builder

from builders.model_builder import get_model_dic_from_file
from builders.model_builder import set_weights, clean_dict
import builders.model_builder as model_builder
from models.trinet import TriNet
from models.classification import Classification
from models.mgn import MGN

from torchvision import transforms
from augmentations import Hflip, ToTensor
from imgaug import augmenters as iaa
import json
from sacred import Experiment
import builders.config_builder as config_builder

from settings import Config

import builders.dataloader_builder as dataloader_builder
import functools
from dataloader import _collate_fn
from torch.utils.data.dataloader import default_collate

transform_cfg = {
    "Resize": {
        "width": 128,
        "height": 256
    }
}
dataset_cfg = {
    "transform": transform_cfg,
    "source_file": Config.MARKET_SOURCE,
    "data_dir": Config.MARKET_DATA,
    "loader_fn": "pil",
    "name": "market1501"
}


dataset_cfgs = [dataset_cfg, dataset_cfg]


coco_stuff_train = {
    'name': "coco_stuff",
    'data_dir': "/work/weber/datasets/COCO",
    'split': "train",
    'source_file': "/work/weber/datasets/COCO/annotations/stuff_train2017.json",
    'transform': {
        "backend": 'torchvision'
    },
}


coco_stuff_val = {
    'data_dir': "/work/weber/datasets/COCO",
    'name': "coco_stuff",
    'loader_fn': 'pil',
    'split': "val",
    'source_file': "/work/weber/datasets/COCO/annotations/stuff_val2017.json",
    'transform': {
        "backend": 'torchvision'
    },
}


@pytest.mark.parametrize("cfg", [coco_stuff_train, coco_stuff_val])
def test_build_coco_stuff(cfg):
    dataset = dataset_builder.build(cfg)
    assert isinstance(dataset, Dataset)

    for datum in dataset:
        seg = datum['gt-seg'].numpy()
        assert len(seg[seg == -1]) == 0

dummy_cfg = {
    "name": "dummy",
    "size": 100,
    "num_pids": 5,
    "id": "short",
    "data_dir": '/'
}
dummy_cfg_long = {
    "name": "dummy",
    "id": "long",
    "size": 200,
    "num_pids": 5,
    "data_dir": '/'
}
def test_seq_sampler():
    config = {
        "type": "Sequential",
        "dataset": dummy_cfg,
        "batch_size": 10
    }
    sampler, _ = sampler_builder.build(config)
    assert isinstance(sampler, BatchSampler)
    assert isinstance(sampler.sampler, SequentialSampler)


def test_pk_sampler():
    config = {
        "type": "pk",
        "P": 10,
        "K": 4,
        "dataset": dummy_cfg
    }
    sampler, _ = sampler_builder.build(config)
    assert isinstance(sampler, PKSampler)


def test_random_sampler_weighted():
    config = {
        "type": "random_sampler_length_weighted",
        "samplers": {
            "sampler1": {
                "type": "pk",
                "dataset": dummy_cfg,
                "P": 10,
                "K": 4
            },
            "sampler2": {
                "type": "pk",
                "dataset": dummy_cfg_long,
                "P": 10,
                "K": 4
            },
        }
    }
    sampler, _ = sampler_builder.build(config)
    assert isinstance(sampler, RandomSamplerLengthWeighted)

@pytest.mark.parametrize("dataset_cfg", [dataset_cfg, dataset_cfgs])
def test_dataset_builder(dataset_cfg):

    sampler_cfg = {
        "type": "pk_sampler",
        "dataset": dataset_cfg,
        "P": 18,
        "K": 4
    }

    sampler, dataset = sampler_builder.build(sampler_cfg)
    assert isinstance(sampler, PKSampler)
    if isinstance(dataset_cfg, list):
        assert isinstance(dataset, ConcatDataset)
    else:
        assert isinstance(dataset, Dataset)


sampler_cfg = {
    "type": "pk_sampler",
    "dataset": dataset_cfg,
    "P": 18,
    "K": 4
}
dataset_cfg2 = {
    "name": "dummy",
    "id": "dataset2",
    "size": 20,
    "num_pids": 5,
    "data_dir": '/'
}
sampler_cfg2 = {
    "type": "pk_sampler",
    "dataset": dataset_cfg2,
    "P": 18,
    "K": 4
}
sampler_cfgs = {
        "type": "concatenated_longest",
        "samplers": {
            "sampler1": sampler_cfg,
            "sampler2": sampler_cfg2
        }
    }


@pytest.mark.parametrize("sampler_cfg", [sampler_cfg, sampler_cfgs])
def test_sampler_builder(sampler_cfg):

    sampler, dataset = sampler_builder.build(sampler_cfg)
    if "samplers" in sampler_cfg:
        assert isinstance(dataset, MultiDataset)
        # TODO they both pass, how to make stricter
        assert isinstance(sampler, MultiSampler)
        assert isinstance(sampler, ConcatenatedSamplerLongest)
    else:
        assert isinstance(sampler.dataset, Dataset)
        assert isinstance(sampler, PKSampler)


batch_hard_cfg  = {
    "type": "BatchHard",
    "name": "BatchHard",
    "margin": "soft",
    "weight": 0.5,
    "dataset": "dummy",
    "endpoint": "test"
}


def test_batch_hard():
    loss = loss_builder.build(batch_hard_cfg)
    assert isinstance(loss, SingleLoss)
    assert isinstance(loss.loss, BatchHard)



multi_loss_cfg = {
        "type": "LinearWeightedLoss",
        "losses": [batch_hard_cfg, batch_hard_cfg]
    }


def test_multi_loss():
    loss = loss_builder.build(multi_loss_cfg)
    assert isinstance(loss, MultiLoss)


def test_datasets_builder():
    dataset = dataset_builder.build(dataset_cfgs)
    assert isinstance(dataset, ConcatDataset)


reid_dataset_cfg = {
    "transform": transform_cfg,
    "source_file": Config.MARKET_SOURCE,
    "data_dir": Config.MARKET_DATA,
    "loader_fn": "pil",
    "name": "market",
    "name": "market1501"
}

reid_dataset_cfg2 = {
    "transform": transform_cfg,
    "source_file": Config.MARKET_SOURCE,
    "data_dir": Config.MARKET_DATA,
    "loader_fn": "pil",
    "name": "reid_dataset2",
    "name": "market1501"
}
def test_reid_dataset_builder():
    dataset = dataset_builder.build(reid_dataset_cfg)
    assert isinstance(dataset, ReidDataset)

reid_dataset_cfgs = [reid_dataset_cfg, reid_dataset_cfg2]
def test_reid_datasets_builder():
    dataset = dataset_builder.build(reid_dataset_cfgs)
    assert isinstance(dataset, ConcatReidDataset)

gallery_cfg = {
    "transform": transform_cfg,
    "source_file": "/home/pfeiffer/Projects/cupsizes/data/market1501_test.csv",
    "data_dir": "/home/pfeiffer/Projects/triplet-reid-pytorch/data/Market-1501",
    "loader_fn": "pil",
    "name": "reid_dataset1",
    "type": "reid"
}

query_cfg = {
    "transform": transform_cfg,
    "source_file": "/home/pfeiffer/Projects/cupsizes/data/market1501_query.csv",
    "data_dir": "/home/pfeiffer/Projects/triplet-reid-pytorch/data/Market-1501",
    "loader_fn": "pil",
    "name": "reid_dataset1",
    "type": "reid"
}

evaluation_cfg = {
        "datasets": {
            "name": "market-1501",
            "gallery": gallery_cfg,
            "query": query_cfg,
            "metric": "euclidean",
            "gallery_batch_size": 10,
            "query_batch_size": 10,
            "num_workers": 4
        },
        "model": {
            "files": ["/work/pfeiffer/master/10/model_300"]
        }
    }
def test_evaluation_builder():
    evaluations, models = evaluation_builder.build(evaluation_cfg)


model_cfg = {
    "name": "trinet",
    "pretrained": True,
    "dim": 256,
    "num_branches": [1, 2, 4],
    "merging_block": {
        "name": "single",
        "endpoint": "triplet"
    }
}

dataset_info = {"num_classes": 751}

models = ["trinet", "classification", "mgn"]
@pytest.mark.parametrize("model_name", models)
def test_model_builder(model_name):
    model_cfg["name"] = model_name
    model_cfg.update(dataset_info)
    model = model_builder.build(model_cfg)
    if model_name == "trinet":
        assert isinstance(model, TriNet)
    elif model_name == "mgn":
        assert isinstance(model, MGN)
    elif model_name == "classification":
        assert isinstance(model, Classification)
    else:
        raise RuntimeError

def test_get_model_dic_from_file():
    path = '/work/pfeiffer/master_old/baseline/final/v1/model_12300'
    dic = get_model_dic_from_file(path)
    print(dic.keys())

 
def test_set_weights():
    path = '/work/pfeiffer/master_old/baseline/final/v1/model_12300'
    dic = get_model_dic_from_file(path)
    model_cfg = {"name": "baseline", "pretrained": False, "backbone": {"name": "resnet", "stride": 1}, "pooling": "max"}
    model = model_builder.build(model_cfg)
    dic = clean_dict(dic)
    model = set_weights(model, dic)
    print(dic.keys())
    
    print(model.state_dict().keys())


def test_duplicate_layer():
    state_dict = {"layer1.0": 1,
                  "layer1.1": 2,
                  "layer2.0": 0
                  }
    to_layers = ["layers3", "layers4"]
    model_builder.duplicate_layer(state_dict, "layer1", to_layers)
    for layer in to_layers:
        assert state_dict[layer+'.0'] == state_dict['layer1.0']
        assert state_dict[layer+'.1'] == state_dict['layer1.1']

def test_filter_weights():
    state_dict = {"layer1.0": 1,
                  "layer1.1": 2,
                  "layer2.0": 0
                  }
    skip = ["layers2"]
    state_dict = model_builder.filter(state_dict, skip)
    for s in skip:
        assert not s in state_dict


transform_cfg = {
    "fliplr": {'p': 0.5},
    "RandomCrop": {
        "width": 128,
        "height": 256,
        "scale": 1.125
    }
}


def test_build_transform():
    arch = transform_builder.build(transform_cfg)
    compose = arch.transform
    for l in compose:
        print(l)
    transform = compose[0]
    assert isinstance(transform, iaa.Fliplr)
    transform = compose[1]
    assert isinstance(transform, iaa.Resize)
    transform = compose[2]
    assert isinstance(transform, iaa.CropToFixedSize)
    assert len(compose) == 3

    assert arch.num_copies == 1

def pretty(dic):
    print(json.dumps(dic, indent=1))

config_files = glob.glob('./configs/*.json')
@pytest.mark.parametrize("config", config_files)
def test_experiment_builder(config):
    ex = Experiment()
    ex = config_builder.build(ex)

    @ex.automain
    def main(_run):
        pretty(_run.config)
    ex.run(named_configs=[config])


mpii_config = {
        "loader_fn": "cv2",
        "name": "mpii",
        "data_dir": Config.MPII_DATA,
        "source_file": Config.MPII_SOURCE,
        "transform": transform_cfg,
        "split": "train"
    }


def test_pose_dataset():
    dataset = dataset_builder.build(mpii_config)
    assert isinstance(dataset, PoseDataset)
    

def test_mpii_dataset():
    dataset = dataset_builder.build(mpii_config)
    assert isinstance(dataset, Mpii)


def test_dataloader_builder():
    cfg = {
        "sampler": sampler_cfgs
    }
    dataloader = dataloader_builder.build(cfg)
    collate_fn1 = dataloader.collate_fn
    assert isinstance(collate_fn1, functools.partial)
    assert collate_fn1.func == _collate_fn
    cfg = {
        "sampler": sampler_cfg
    }
    dataloader = dataloader_builder.build(cfg)
    collate_fn2 = dataloader.collate_fn
    # TODO
    assert collate_fn2 == default_collate
