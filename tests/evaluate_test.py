from torch.nn.parallel import DataParallel
import builders.evaluation_builder as evaluation_builder
import builders.dataloader_builder as dataloader_builder
import builders.model_builder as model_builder
import h5py
from metrics import calculate_pckh
from calculate_score import score
import torch
from datasets import mpii
from settings import Config

transform_cfg = {
    "backend": "torchvision",
    "resize": {
        "width": 128,
        "height": 256
    },
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std" : [0.229, 0.224, 0.225]
    },
    "augmentations": ["HorizontalFlip"]
}

reid_cfg = {
    "sampler": {
        "type": "sequential",
        "dataset": {
            "name": "market1501_test",
            "data_dir": Config.MARKET_DATA,
            "test": Config.MARKET_TEST,
            "query": Config.MARKET_QUERY,
            "transform": transform_cfg,
            "metric": "euclidean",
            "loader_fn": "cv2pil"
        },
        "batch_size": 10,
        "drop_last": False
    }
}

baseline_model_cfg = {
    "name": "baseline",
    "backbone": {
        "name": "ResNet",
        "stride": 1
    },
    "pooling": "max",
    "files": ["/work/pfeiffer/master/baseline/278/model_300"]
}

baseline_model_cfg = {
    "files": ["/globalwork/pfeiffer/master/architecture_sampling_pose_reid_v2/419/model_300"]
}

from main import evaluate
def test_reid_evaluation():
    dataloader = dataloader_builder.build(reid_cfg)
    # restore
    model_cfgs = evaluation_model_builder.build(baseline_model_cfg)

    model = model_builder.build(model_cfgs[0])
    model = DataParallel(model)
    _run = {'config': {'device': torch.device('cuda')}}
    score = evaluate([dataloader], model, _run, "test")

    print(score)


test_dataset_cfg = {
    "transform": transform_cfg,
    "source_file": "/home/pfeiffer/Projects/cupsizes/data/market1501_test.csv",
    "data_dir": "/home/pfeiffer/Projects/triplet-reid-pytorch/data/Market-1501",
    "loader_fn": "pil",
    "filter_fn": "filter_junk",
    "name": "reid_dataset1",
    "type": "reid"
    }

gt_dataset_cfg = {
    "type": "attribute",
    "dataset_fn": "make_market_attribute",
    "kwargs": {
        "split": "test"
    },
    "source_file": "/home/pfeiffer/Projects/triplet-reid-pytorch/data/Market-1501_Attribute/market_attribute.mat"
}

attributes = ['gender', 'hair', 'up', 'down', 'clothes', 'hat', 'backpack', 'bag', 'handbag', 'age', 'upcolor', 'downcolor']
attribute_test_dataset_cfg = {
    "name": "market-1501-attribute",
    "attributes": attributes,
    "test_dataset": test_dataset_cfg,
    "attribute_dataset": gt_dataset_cfg,
    "batch_size": 10,
    "num_workers": 0
}

attribute_model_cfg = {
        "files": ["/work/pfeiffer/master/attributes/57/model_300"]
    }

attribute_evaluation_cfg = {
    "datasets": attribute_test_dataset_cfg,
    "model": attribute_model_cfg,
    "delete": True
}
def test_score():
    score('/tmp/tmp1xwnn43w_home_pfeiffer_Projects_cupsizes_data_market1501_query.csv', '/tmp/tmp2ofy8680_home_pfeiffer_Projects_cupsizes_data_market1501_test.csv', 10, 'market', 'euclidean')


def test_attribute_evaluation():
    evaluation, model_cfgs = evaluation_builder.build(attribute_evaluation_cfg)
    with torch.no_grad():
        for model_cfg in model_cfgs:
            model = model_builder.build(model_cfg)
            model = DataParallel(model)
            score = evaluate(evaluation, model, delete=True)
            print(score)

reid_evaluation_cfg = {}
reid_attribute_evaluation_cfg = {
    "datasets": [attribute_test_dataset_cfg, reid_evaluation_cfg],
    "model": attribute_model_cfg,
    "delete": True
}
from main import format_result

def test_reid_attribute_evaluation():
    #result = run_evaluation(reid_attribute_evaluation_cfg)
    #formatted_result = format_result(result)
    #print('result', formatted_result)
    pass

from datasets.mpii import make_dataset as make_mpii
import numpy as np


def generate_fake_2d_joints_from_original(data, deviation):
    random = np.random.rand(*data.shape) * 2 - 1
    random *= deviation[:, None, None]
    random += data
    return random

pose_dataset_cfg = {
    "sampler": {
        "type": "sequential",
        "dataset": {
            "name": "mpii",
            "source_file": Config.MPII_SOURCE,
            "data_dir": Config.MPII_DATA,
            "split": "val",
            "loader_fn": "cv2",
            "transform": {
                "cropfrombbox": {},
                "resize": {
                    "width": 256,
                    "height": 256
                },
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std" : [0.229, 0.224, 0.225]
                }
            }
        },
        "batch_size": 10
    }
}

pose_model_cfg = {
    "files": ["/globalwork/pfeiffer/master/architecture_sampling_pose_reid_v2/419/model_300"]
}

from builders import (dataset_builder, evaluation_model_builder)
import os

def test_pose_evaluation():
    dataloader = dataloader_builder.build(pose_dataset_cfg)
    # restore
    model_cfgs = evaluation_model_builder.build(pose_model_cfg)

    model = model_builder.build(model_cfgs[0])
    model = DataParallel(model)
    _run = {'config': {'device': torch.device('cuda')}}
    score = evaluate([dataloader], model, _run, "test")

    print(score)

def test_pose_writing():
    pose_dataset = dataset_builder.build(pose_dataset_cfg)
    pose_dataloader = torch.utils.data.DataLoader(
            pose_dataset,
            batch_size=10,
            num_workers=0
    )
    model_cfg = evaluation_model_builder.build(pose_model_cfg)
    model = model_builder.build(model_cfg[0])
    model = DataParallel(model)
    filename = "tests/pose.h5"
    if os.path.isfile(filename):
        os.remove(filename)
        print("deleted old {}".format(filename))
    write_to_h5(pose_dataloader, model, "tests/pose.h5", ["pose"])


def compute_pck(dist, pck_th_range):
    """https://github.com/NieXC/pytorch-mula/blob/master/utils/calc_pckh.py

    Only used for verification.
    """

    P = dist.shape[1]
    pck = np.zeros([len(pck_th_range), P + 2])

    # For individual joint
    for p in range(P):
        for thi in range(len(pck_th_range)):
            th = pck_th_range[thi]
            joint_dist = dist[:, p]
            pck[thi, p] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

    # For uppper body
    for thi in range(len(pck_th_range)):
        th = pck_th_range[thi]
        joint_dist = dist[:, 8:16]
        pck[thi, P] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

    # For all joints
    for thi in range(len(pck_th_range)):
        th = pck_th_range[thi]
        joints_index = list(range(0,6)) + list(range(8,16))
        joint_dist = dist[:, joints_index]
        pck[thi, P + 1] = 100 * np.mean(joint_dist[np.where(joint_dist >= 0)] <= th)

    return pck

def pck_table_output_lip_dataset(pck, method_name):
    str_template = '{0:10} & {1:6} & {2:6} & {3:6} & {4:6} & {5:6} & {6:6} & {7:6} & {8:6} & {9:6}'
    head_str = str_template.format('PCKh@0.5', 'Head', 'Sho.', 'Elb.', 'Wri.', 'Hip', 'Knee', 'Ank.', 'U.Body', 'Avg.')
    num_str = str_template.format(method_name, '%1.5f'%((pck[8]  + pck[9])  / 2.0),
                                               '%1.5f'%((pck[12] + pck[13]) / 2.0),
                                               '%1.5f'%((pck[11] + pck[14]) / 2.0),
                                               '%1.5f'%((pck[10] + pck[15]) / 2.0),
                                               '%1.5f'%((pck[2]  + pck[3])  / 2.0),
                                               '%1.5f'%((pck[1]  + pck[4])  / 2.0),
                                               '%1.5f'%((pck[0]  + pck[5])  / 2.0),
                                               '%1.5f'%(pck[-2]),
                                               '%1.5f'%(pck[-1]))
    print(head_str)
    print(num_str)

def test_verify_pckh():
    with h5py.File('./tests/pose.h5', 'r') as f:
        bbox = to_float(np.asarray(f['bbox']))
        left = bbox[:, 0]
        top = bbox[:, 1]
        right = bbox[:, 2]
        bottom = bbox[:, 3]
        width = right - left
        height = bottom - top

        data = np.asarray(f['pose'])
        data = np.squeeze(data)
        data[:, :, 0] *= width[:, None]
        data[:, :, 1] *= height[:, None]

        gt_data = np.asarray(f['coords'])
        gt_data = to_float(gt_data)
        gt_data[:, :, 0] *= width[:, None]
        gt_data[:, :, 1] *= height[:, None]

        head_sizes = np.asarray(f['head_size'])
        head_sizes = to_float(head_sizes)
        dist = np.linalg.norm(gt_data - data, axis=2)
        dist /= head_sizes[:, None]
        pck_th_range = [0.1, 0.5]
        pck = compute_pck(dist, pck_th_range)
        # NOTE there is a small difference in overall pckh
        pck_table_output_lip_dataset(pck[0], "0.1")
        pck_table_output_lip_dataset(pck[1], "0.5")

def test_compare_pckh():
    test_mpii_evaluation()
    test_verify_pckh()


def to_float(array):
    return array.astype(np.float)


def test_mpii_evaluation():

    score = mpii.Mpii.evaluate('./tests/pose.h5', mpii.make_joint_info())
    print(score)

def test_pckh_evaluation():
    with h5py.File('./tests/pose.h5', 'r') as f:
        data = np.asarray(f['pose']) * 256
        gt_data = np.asarray(f['coords'])
        gt_data = to_float(gt_data) * 256
        head_sizes = np.asarray(f['head_size'])
        head_sizes = to_float(head_sizes)
        print(gt_data.shape)
        print(data.shape)
        data = np.squeeze(data)
        pck, pck_joint = calculate_pckh(gt_data, data, head_sizes)
        print(pck, pck_joint)
