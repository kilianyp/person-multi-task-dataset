from PIL import Image
import numpy as np
import builders.transform_builder as transform_builder
from augmentations import normalize
from augmentations import Normalize
import imgaug as ia
from datasets.utils import cv2_loader
from utils import visualize, to_cv
from datasets.pose_dataset import PoseDataset
from datasets.mpii import make_dataset as make_mpii
import cv2
from torchvision import transforms
from augmentations import ToTensor
from datasets.utils import pil_loader, pil_2_cv_loader
np.set_printoptions(precision=3)


def make_random_2d_keypoints(num, img_shape):
    img_shape = np.asarray(img_shape)
    print(img_shape)
    keypoints = np.random.rand(num, 2)
    for keypoint in keypoints:
        factor = np.random.rand(2) * img_shape
        keypoint *= factor
    
    return keypoints


def make_edge_case_keypoints(shape):
    return np.asarray([[0, 0], [shape[0], shape[1]]])


source_file = '/work/pfeiffer/datasets/mpii/mpii_human_pose_v1_u12_1.mat'
data_dir = '/work/pfeiffer/datasets/mpii/'
def test_random_crop_transform():
    transform_cfg = {
        "RandomHorizontalFlipWithPairs": {'p': 0.5},
        "RandomCrop": {
            "width": 256,
            "height": 384,
            "scale": 1.125
        },
        "debug": True
    }
    data, header, dataset_info = make_mpii(source_file, data_dir)
    joint_info = dataset_info['joint_info']
    transform = transform_builder.build(transform_cfg)
    pose_dataset = PoseDataset(data, header, {}, cv2_loader, 'mpii', transform=transform)
    for _ in range(4):
        idx = np.random.randint(0, len(pose_dataset))
        data = pose_dataset[idx]
        keypoints = data['coords']
        image = data['img']
        visualize(image, keypoints, joint_info.stick_figure_edges)
    # (batch, height, width, channels)
    # This is just imgaug augmentation
    # pytorch will convert again
    assert image.shape == (384, 256, 3)
    assert not np.any(keypoints > 384*1.125)


def test_keypoints_normalize():
    shape = (640, 360)
    keypoints = make_random_2d_keypoints(10, shape)
    edge_cases = make_edge_case_keypoints(shape)
    keypoints = np.concatenate([keypoints, edge_cases])
    keypoints = ia.KeypointsOnImage.from_coords_array(keypoints, shape)
    norm_transform = Normalize(width=640, height=360)
    keypoints_aug = norm_transform.augment_keypoints([keypoints])
    keypoints_aug = keypoints_aug[0].get_coords_array()
    assert not np.any(keypoints_aug > 1.0)
    assert not np.any(keypoints_aug < -1.0)


def test_keypoints_normalize_with_scale():
    transform_cfg = {
        "resize": {
            "width": 256,
            "height": 384
        }
    }
    shape = (640, 360)
    normalize = Normalize(0, 0, height=384, width=256)
    transform = transform_builder.build(transform_cfg)
    transform.append(normalize)
    # points are x,y
    keypoints = make_random_2d_keypoints(10, shape)
    edge_cases = make_edge_case_keypoints(shape)
    keypoints = np.concatenate([keypoints, edge_cases])
    # shape is expected height, width
    keypoints = ia.KeypointsOnImage.from_coords_array(keypoints, (360, 640))
    keypoints_aug = transform.augment_keypoints([keypoints])
    keypoints_aug = keypoints_aug[0].get_coords_array()
    assert not np.any(keypoints_aug > 1.0)
    assert not np.any(keypoints_aug < -1.0)


def test_resize_keypoints():
    transform_cfg = {
        "resize": {
            "width": 256,
            "height": 384
        }
    }
    shape = (360, 640)
    image = to_cv(Image.new("RGB", shape))
    data, header, dataset_info = make_mpii(source_file, data_dir)
    transform = transform_builder.build(transform_cfg)
    print(transform)
    joint_info = dataset_info['joint_info']
    keypoints = make_random_2d_keypoints(14, shape)
    # add edge cases
    edge_cases = np.asarray([[0, 0], [shape[0], shape[1]]])
    keypoints = np.concatenate([keypoints, edge_cases])
    print(keypoints)
    print(image.shape)
    keypoints = ia.KeypointsOnImage.from_coords_array(keypoints, image.shape)
    image = transform.augment_image(image)
    keypoints_aug = transform.augment_keypoints([keypoints])
    keypoints = keypoints_aug[0].get_coords_array()

    visualize(image, keypoints, joint_info.stick_figure_edges)
    # print(keypoints)
    # (batch, height, width, channels)
    # This is just imgaug augmentation
    # pytorch will convert again
    print(keypoints)
    assert image.shape == (384, 256, 3)
    assert not np.any(keypoints[:, 0] > 256)
    assert not np.any(keypoints[:, 1] > 384)


def test_to_tensor():
    path = '/work/pfeiffer/datasets/mpii/images/000022704.jpg'

    #cv2_image = cv2_loader(path)
    pil_image  = pil_loader(path)
    cv2_image = np.asarray(pil_image)
    torch_transforms = [transforms.ToTensor()]
    compose = transforms.Compose(torch_transforms)
    valid = compose(pil_image)


    to_tensor = ToTensor()
    image_aug = to_tensor.augment_image(cv2_image)
    print(image_aug.shape)
    assert image_aug.shape == valid.shape
    np.testing.assert_array_almost_equal(valid.data.numpy()[0, ...], image_aug.data.numpy()[0,...], decimal=3)


def test_normalization():
    path = '/work/pfeiffer/datasets/mpii/images/000022704.jpg'

    pil_image  = pil_loader(path)
    print(pil_image.mode)
    pil_image = pil_image.convert('RGB')
    print(pil_image.mode)
    cv2_image = pil_2_cv_loader(path)

    mean = [0.54, 0.51, 0.25]
    std = [0.112, 0.154, 0.351]
    normalize = transforms.Normalize(mean, std)
    torch_transforms = [transforms.ToTensor(), normalize]
    compose = transforms.Compose(torch_transforms)
    valid = compose(pil_image)

    to_tensor = ToTensor()
    image_aug = to_tensor.augment_image(cv2_image)
    image_aug = normalize(image_aug)
    assert image_aug.shape == valid.shape
    np.testing.assert_array_almost_equal(valid.data.numpy(), image_aug.data.numpy(), decimal=3)
