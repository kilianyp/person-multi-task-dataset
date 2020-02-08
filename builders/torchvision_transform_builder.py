import torch
from PIL import Image
import torchvision.transforms as transforms
from transform import TorchvisionTransform
from augmentations import Hflip, no_augmentation
import numpy as np
import logger as log
logger = log.get_logger()

def build(cfg, dataset_info):
    """
    Builds an image transformation.

    Differs between transform and augmentations.
    Augmentations returns more than one image.
    """

    trans = []
    for transform_name, transform_cfg in cfg.items():
        transform = _build_transform(transform_name, transform_cfg, dataset_info)
        if transform is not None:
            trans.extend(transform)

    transform = transforms.Compose(trans)

    augmentation_names = cfg.get('augmentations', [])
    if len(augmentation_names) > 1:
        raise NotImplementedError('More than one augmentation is not supported')

    to_tensor = transforms.ToTensor()
    norm_cfg = cfg.get('normalization', {})
    normalization = build_normalization(norm_cfg)

    def to_normalized_tensor(img):
        return normalization(to_tensor(img))

    num_copies = 1
    if len(augmentation_names) != 0:
        augmentation_fn, num_copies = _build_augmentation(augmentation_names[0], cfg)
        trans.append(augmentation_fn)
        normalize = transforms.Lambda(lambda crops: torch.stack([to_normalized_tensor(crop) for crop in crops]))
    else:
        normalize = to_normalized_tensor


    debug = cfg.get('debug', False)
    torchvision_transform = TorchvisionTransform(transform, normalize, debug, num_copies=num_copies)
    return torchvision_transform


def build_normalization(cfg):
    """
    TODO currently the same for both backends.
    Is used in dataset builder.
    make normalization model specific?
    """
    if 'mean' in cfg:
        mean = cfg['mean']
    else:
        logger.warning('No mean for nomalization given!')
        mean = np.zeros(3)

    if 'std' in cfg:
        std = cfg['std']
    else:
        logger.warning('No std for nomalization given!')
        std = np.ones(3)

    return transforms.Normalize(mean, std)


def _build_transform(name, cfg, dataset_info):
    ignore = ['augmentations', 'debug', 'backend', 'normalization']
    name = name.lower()
    if name == 'randomhorizontalflip':
        return [transforms.RandomHorizontalFlip()]
    elif name == 'resize':
        H = cfg['height']
        W = cfg['width']
        # cv2 uses BICUBIC as default, torchvision Linear
        interpolation = cfg.get('interpolation', Image.BICUBIC)
        return [transforms.Resize((H, W), interpolation)]
    elif name == 'randomcrop':
        H = cfg['height']
        W = cfg['width']
        return [transforms.RandomCrop((H, W))]
    elif name in ignore:
        return None


    raise ValueError("Unkown transform {}.".format(name))


def _build_augmentation(name, cfg):
    """
    Returns: Tuple of the augmentation function and the number of replications it will create."""
    name = name.lower()
    if name == "tencrop":
        return (transforms.TenCrop((cfg['height'], cfg['width'])), 10)
    elif name == "horizontalflip":
        return (Hflip(), 2)
    elif name == "no_augmentation":
        return (no_augmentation, 1)
    raise NotImplementedError("Augmentation does not exist, choices from: {}".format(augmentation_choices.keys()))
