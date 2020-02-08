from datasets.pose_dataset import build_mapping_pairs
from datasets.utils import pil_loader, cv2_loader
from transforms.flip_lr_with_pairs import FliplrWithPairs
from datasets.pose_dataset import PoseDataset
from datasets.mpii import make_dataset as make_mpii
from utils import visualize
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
from utils import to_cv
from imgaug import parameters as iap
import torch
from models.pose import SoftArgMax2d
import pytest

from augmentations import Crop
from builders import dataset_builder



def test_imgaug():
    data, headers, dataset_info = make_mpii()
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


def test_flipping():
    data, headers, dataset_info = make_mpii()
    joint_info = dataset_info['joint_info']
    ia.seed(1)
    seq = FliplrWithPairs(1)
    seq.pairs = joint_info.mirror_mapping_pairs
    for d in data[:5]:
        image = np.asarray(Image.open(d['path']))
        coords = d['coords']
        keypoints = ia.KeypointsOnImage.from_coords_array(coords, image.shape)
        visualize(to_cv(image), keypoints.get_coords_array(), joint_info.stick_figure_edges)
        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
        visualize(to_cv(image_aug), keypoints_aug.get_coords_array(), joint_info.stick_figure_edges)


def test_build_mapping_pairs():
    ids = {"l_arm": 0, "r_arm": 1, "l_ankle": 2, "r_ankle": 3}
    
    mapping_pairs = build_mapping_pairs(ids)

    print(mapping_pairs)
    assert len(mapping_pairs) == 2
    assert mapping_pairs[0] == (0, 1)
    assert mapping_pairs[1] == (2, 3)

source_file = '/work/pfeiffer/datastets/mpii/mpii_human_pose_v1_u12_1.mat'
data_dir = '/work/pfeiffer/datasets/mpii_human_pose_v1/'

import pytest
@pytest.mark.xfail()
def test_augmentations():
    data, header, joint_info = make_mpii(source_file, data_dir)
    transform = FliplrWithPairs(joint_info.mirror_mapping_pairs, 0.5)
    pose_dataset = PoseDataset(data, header, cv2_loader, 'mpii', transform=transform)
    coords1 = pose_dataset[25]['coords']
    # there is a possibility this never fails
    for _ in range(100):
        data = pose_dataset[25]
        np.testing.assert_equal(coords1, data['coords'])

def test_flip_augmentation():
    data, header, dataset_info = make_mpii(source_file, data_dir)
    joint_info = dataset_info['joint_info']
    transform = FliplrWithPairs(joint_info.mirror_mapping_pairs, 0.5)
    transform.debug = True
    transform.augmentation = None
    pose_dataset = PoseDataset(data, header, dataset_info, cv2_loader, 'mpii', transform=transform)
    for _ in range(10):
        data = pose_dataset[25]
        visualize(to_cv(data['img']), data['coords'], joint_info.stick_figure_edges)


def test_bbox_crop():
    pose_dataset, header, dataset_info = make_mpii(source_file, data_dir)
    joint_info = dataset_info['joint_info']
    transform = Crop()
    for _ in range(10):
        idx = np.random.randint(0, len(pose_dataset))
        data = pose_dataset[idx]
        img = cv2_loader(data['path'])
        print(img.shape)
        bbox = ia.BoundingBoxesOnImage.from_xyxy_array(data['bbox'][None,:], shape=img.shape)
        img = transform.augment_image(img, bbox.bounding_boxes[0])
        keypoints = ia.KeypointsOnImage.from_coords_array(data['coords'], img.shape)
        keypoints = transform.augment_keypoints(keypoints, bbox.bounding_boxes[0])

        coords = keypoints.get_coords_array()
        visualize(img, coords, joint_info.stick_figure_edges)


mpii_config = {
        "name": "mpii",
        "source_file": "/work/pfeiffer/datasets/mpii/mpii_human_pose_v1_u12_1.mat",
        "data_dir": "/work/pfeiffer/datasets/mpii/",
        "dataset_fn": "mpii",
        "loader_fn": "cv2",
        "transform": {
            "affine": {
                "translate_percent": [-0.02, 0.02],
                "scale": [0.75, 1.25]
            },
            "randomhorizontalflip": {"p": 0.5},
            "resize": {
                "width": 256,
                "height": 256
            },
            "debug": True
        },
        "width": 256,
        "height": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "type": "pose",
    }

def test_transform():
    dataset = dataset_builder.build(mpii_config)
    joint_info = dataset.info['joint_info']
    for _ in range(10):
        idx = np.random.randint(0, len(dataset))
        #idx = 15707
        data = dataset[idx]
        img = data['img']
        coords = data['coords']
        #print("after to tensor", img)
        visualize(img, coords, joint_info.stick_figure_edges)


@pytest.mark.parametrize("num_joints, height, width", [(2, 2, 2), (16, 8, 4)])
def test_softargmax_simple(num_joints, height, width):
    batch_size = 4
    x = torch.zeros(batch_size, num_joints, height, width)
    soft = SoftArgMax2d()
    result = soft(x)
    assert result.shape == (batch_size, num_joints, 2)
    np.testing.assert_array_equal(result.data.numpy(), np.ones((batch_size, num_joints, 2)) * 0.5)
    x = torch.rand(batch_size, num_joints, height, width)
    result = soft(x)


def test_softargmax_advanced():
    # only works for 2x2
    # sums to 100
    input = [[10, 3], [5, 11]]
    input = np.array(input, dtype=np.float32)
    shape = input.shape
    soft_max_test_in = np.reshape(input, -1)
    ex = np.exp(soft_max_test_in)
    ex /= ex.sum()
    ex = ex.reshape(shape)
    correct_result = (np.sum(ex[:, 1]), np.sum(ex[1, :]))

    input = input[None, None, :, :]
    input = torch.from_numpy(input)
    soft = SoftArgMax2d()
    result = soft(input)
    result = result.data.numpy()
    np.testing.assert_array_almost_equal(result[0, 0, :], correct_result)
