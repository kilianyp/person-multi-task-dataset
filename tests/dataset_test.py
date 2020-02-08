from datasets.dataset import ConcatDataset, MultiDataset, unique_header
from builders.dataloader_builder import build_collate_fn, ERROR_STRING
from torch.utils.data import DataLoader
from datasets.utils import make_dataset_default, cv2_loader
from samplers.sequential_sampler import SequentialSampler
from samplers.batch_sampler import BatchSampler
from samplers.multi_sampler import ConcatenatedSamplerLongest
import pytest
from datasets.dummy import create_dummy_data, create_dummy_pid_data, DummyDataset
import torch
from datasets.reid_dataset import rewrite_pids, ReidDataset, ConcatReidDataset
from datasets.attribute_dataset import AttributeReidDataset, AttributeDataset
from datasets.mpii import make_dataset as make_mpii
from utils import visualize
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
import builders.dataset_builder as dataset_builder
from datasets.attribute.market import make_market_attribute
from datasets.attribute.duke_mtmc import make_duke_attribute

from augmentations import ToTensor
from settings import Config

from builders import dataloader_builder



market_config = {
    "name": "market1501",
    "source_file": Config.MARKET_SOURCE,
    "data_dir": Config.MARKET_DATA,
    "loader_fn": "cv2",
    "transform": {
        "resize": {
            "width": 256,
            "height": 256
        },
        "debug": True

    }
}


dataloader_cfg = {
    "sampler": {
        "type": "sequential",
        "dataset": market_config,
        "batch_size": 1
    }
}


def test_dataset_simple():
    dataloader = dataloader_builder.build(dataloader_cfg)
    for idx, data in enumerate(dataloader):
        assert 'img' in data
        assert 'path' in data

        assert isinstance(data['img'], torch.Tensor)
        assert data['path'] != ERROR_STRING
        if idx > 500:
            break


def test_multi_dataset():
    size1 = 70
    size2 = 100
    dummy_cfg_small = {
        "name": "dummy",
        "id": "dummy_small",
        "size": size1,
        "data_dir": "/"
    }

    dummy_cfg_large = {
        "name": "dummy",
        "id": "dummy_large",
        "size": size2,
        "data_dir": "/"
    }

    sequential_cfg1 = {
        "type": "sequential",
        "dataset": dummy_cfg_small,
        "batch_size": 1,
        "drop_last": True
    }

    sequential_cfg2 = {
        "type": "sequential",
        "dataset": dummy_cfg_large,
        "batch_size": 1,
        "drop_last": True
    }

    sampler_cfg = {
        "type": "concatenated_longest",
        "samplers": {
            "sampler1": sequential_cfg1,
            "sampler2": sequential_cfg2
        }
    }

    dataloader_cfg = {
        "sampler": sampler_cfg
    }

    dataloader = dataloader_builder.build(dataloader_cfg)

    for idx, data in enumerate(dataloader):
        assert data['path'][0].startswith("dummy_small")
        assert data['path'][1].startswith("dummy_large")

    test =  size1 if size1 > size2 else size2
    print(test, idx)


def test_concat_dataset():
    size1 = 70
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    dataset1 = DummyDataset(lambda: create_dummy_pid_data(size1, 30, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_data(size2, name2), name2)

    dataset = ConcatDataset([dataset1, dataset2])
    assert len(dataset) == size1 + size2

    sampler = SequentialSampler(dataset)

    collate_fn = build_collate_fn(dataset.header)

    dataloader = DataLoader(
            dataset,
            sampler=sampler,
            num_workers=1,
            collate_fn=collate_fn
            )

    for idx, data in enumerate(dataloader):
        if idx < size1:
            # returns seq samplerbatch of 1
            assert data['path'][0].startswith(name1)
            assert data['pid'][0] != -1
        else:
            assert data['path'][0].startswith(name2)
            assert data['pid'][0] == -1


def test_unique_headers():
    class HeaderDataset(object):
        def __init__(self, header):
            self.header = header

    header1 = HeaderDataset({'test': 1})
    header2 = HeaderDataset({'test': 1})

    header = unique_header([header1, header2])
    assert type(header) == dict
    assert header['test'] == 1

    header3 = HeaderDataset({'test': 2})
    with pytest.raises(RuntimeError):
        header = unique_header([header1, header3])


def test_concat_reid_dataset():
    size1 = 70
    size2 = 100
    name1 = "Dummy1"
    name2 = "Dummy2"
    pid1 = 30
    pid2 = 30
    dataset1 = DummyDataset(lambda: create_dummy_pid_data(size1, pid1, name1), name1)
    dataset2 = DummyDataset(lambda: create_dummy_pid_data(size2, pid2, name2), name2)
    dataset = ConcatReidDataset([dataset1, dataset2])
    assert dataset.num_labels == pid1 + pid2


def test_rewrite_pids():
    d1 = {'pid': 'a'}
    d2 = {'pid': 'b'}
    d3 = {'pid': 'c'}
    d4 = {'pid': 'a'}
    data = [d1, d2, d3, d4]
    num_labels, label_dic = rewrite_pids(data)

    assert num_labels == 3
    assert d4['pid'] == 0


def test_make_market_attribute_train():
    data, headers, dataset_info = make_market_attribute(Config.MARKET_ATTRIBUTE, "train")
    assert len(data) == 751


def test_market_attribute_dataset():
    market_attribute_cfg = {
        "data_dir": Config.MARKET_ATTRIBUTE,
        "split": 'train',
        'name': 'market1501_attribute'
    }

    data = dataset_builder.build(market_attribute_cfg)

    assert data[0]['hat'] == 0
    assert data[174]['hat'] == 1
    assert data[0]['upcolor'] == 2
    assert data[0]['downcolor'] == 6
    for idx, d in enumerate(data):
        assert(d['downcolor']) != 9, idx


def test_make_market_attribute_gallery():
    data, headers, dataset_info = make_market_attribute(Config.MARKET_ATTRIBUTE, "train")
    assert len(data) == 751


def test_make_duke_attribute_gallery():
    data, headers, dataset_info = make_duke_attribute(Config.DUKE_ATTRIBUTE, "train")
    assert len(data) == 702


def test_duke_attribute_dataset():
    duke_attribute_cfg = {
        "data_dir": Config.DUKE_ATTRIBUTE,
        "split": 'train',
        'name': 'duke_mtmc_attribute'
    }

    data = dataset_builder.build(duke_attribute_cfg)

    assert data[0]['hat'] == 0
    assert data[4]['hat'] == 1
    assert data[7]['upcolor'] == 5
    assert data[336]['downcolor'] == 5
    for idx, d in enumerate(data):
        assert(d['gender']) < 2, idx
        assert(d['top']) < 2, idx
        assert(d['boots']) < 2, idx
        assert(d['hat']) < 2, idx
        assert(d['backpack']) < 2, idx
        assert(d['bag']) < 2, idx
        assert(d['handbag']) < 2, idx
        assert(d['shoes']) < 2, idx
        assert(d['upcolor']) < 8, idx
        assert(d['downcolor']) < 7, idx




def test_make_mpii():
    data, headers, dataset_info = make_mpii(Config.MPII_SOURCE, Config.MPII_DATA, "mpii")


def test_viz():
    data, _, dataset_info = make_mpii(Config.MPII_SOURCE, Config.MPII_DATA, "mpii")
    joint_info = dataset_info['joint_info']
    for d in data[:5]:
        coords = d['coords']
        # find top left
        top_x = 9999
        top_y = 9999
        bottom_x = 0
        bottom_y = 0
        for coord in coords:
            x, y = coord
            if x < top_x:
                top_x = x
            if x > bottom_x:
                bottom_x = x
            if y < top_y:
                top_y = y
            if y > bottom_y:
                bottom_y = y

        bbox = [(top_x, top_y), (bottom_x, bottom_y)]


        visualize(d['path'], d['coords'], joint_info.stick_figure_edges, bbox)


def test_pose_imgaug():
    data, headers, dataset_info = make_mpii(Config.MPII_SOURCE, Config.MPII_DATA, "mpii")
    joint_info = dataset_info['joint_info']
    ia.seed(1)
    seq = iaa.Sequential([
        iaa.Affine(
            rotate=10,
            scale=(0.5, 0.7)
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])
    for d in data[:5]:
        image = np.asarray(Image.open(d['path']))
        coords = d['coords']
        keypoints = ia.KeypointsOnImage.from_coords_array(coords, image.shape)
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
        open_cv_image = np.array(image_aug)
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        visualize(open_cv_image, keypoints_aug.get_coords_array(), joint_info.stick_figure_edges)


transform_cfg = {
    "RandomHorizontalFlipWithPairs": {'p': 0.5},
    "RandomCrop": {
        "width": 128,
        "height": 256,
        "scale": 1.125
    }
}


mpii_config = {
        "name": "mpii",
        "split": "train",
        "source_file": Config.MPII_SOURCE,
        "data_dir": Config.MPII_DATA,
        "loader_fn": "cv2",
        "transform": {
            "affinewithcrop": {
                "translate_percent": [-0.02, 0.02],
                "scale": [0.75, 1.25]
            },
            "fliplrwithpairs": {"p": 0.5},
            "resize": {
                "width": 256,
                "height": 256
            }

        },
        "width": 256,
        "height": 256,
        "debug": True
    }


def test_pose_dataset():
    dataset = dataset_builder.build(mpii_config)
    for data in dataset:
        assert data['img'].shape == (3, 256, 256)
        print(data['coords'])
        break
