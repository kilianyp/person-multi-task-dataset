from imgaug import augmenters as iaa
from imgaug.imgaug import imresize_single_image
from transforms.affine_with_crop import AffineWithCrop
from transforms.flip_lr_with_pairs import FliplrWithPairs
from transforms.crop_from_bbox import CropFromBbox
from transform import ImgAugTransform, ImgAugTransformWithCrop
import torchvision.transforms as transforms
import torch.tensor
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

    transform = iaa.Sequential(trans)


    augmentation_names = cfg.get('augmentations', [])
    if len(augmentation_names) > 0:
        raise NotImplementedError('Test time augmentations are currently'
                                  'not supported for imgaug')

    # cannot use sequential because it checks the type
    # TODO do normalize on cv image
    to_tensor = transforms.ToTensor()
    norm_cfg = cfg.get('normalization', {})
    normalization = build_normalization(norm_cfg)

    def to_normalized_tensor(img):
        return normalization(to_tensor(img))

    # check if crop from bbox transform is used
    crop_trans = None
    for t in trans:
        if isinstance(t, (AffineWithCrop, CropFromBbox)):
            if crop_trans is not None:
                raise RuntimeError("Two Crop from BBox augmentations are used!")
            crop_trans = t

    debug = cfg.get('debug', False)
    if crop_trans is not None:
        return ImgAugTransformWithCrop(crop_trans, transform, to_normalized_tensor, debug)
    else:
        return ImgAugTransform(transform, to_normalized_tensor, debug)


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


def conv2tuple(l):
    if isinstance(l, list):
        return tuple(l)
    return l


def _build_transform(name, cfg, dataset_info):
    # TODO handle lists and tuples
    # for now all lists
    # list: one value is selected
    # tuple: a value in between the two values is selected
    # TODO handle unknown parameters
    ignore = ['augmentations', 'debug', 'backend', 'normalization']
    name = name.lower()
    if name == 'fliplr':
        return [iaa.Fliplr(cfg.get('p', 0.5))]
    elif name == "fliplrwithpairs":
        if 'joint_info' in dataset_info:
            joint_info = dataset_info['joint_info']
            keypoint_pairs = joint_info.mirror_mapping_pairs
        else:
            keypoint_pairs = None

        if 'seg_info' in dataset_info:
            seg_info = dataset_info['seg_info']
            segmentation_pairs = seg_info.pairs
        else:
            segmentation_pairs = None

        return [FliplrWithPairs(cfg.get('p', 0.5), keypoint_pairs=keypoint_pairs,
                                segmentation_pairs=segmentation_pairs)]
    elif name == 'resize':
        H = cfg['height']
        W = cfg['width']
        interpolation = cfg.get('interpolation', 'cubic')
        return [iaa.Resize({"height": int(H), "width": int(W)},
                          interpolation=interpolation, name='resize')]
    elif name == 'affine':
        rotate = cfg.get('rotate', 0)
        translate_percent = cfg.get('translate_percent')
        translate_px = cfg.get('translate_px')
        shear = cfg.get('shear', 0.0)
        scale = cfg.get('scale', 1.0)
        return [iaa.Affine(scale, translate_percent, translate_px, rotate, shear)]
    elif name == 'affinewithcrop':
        rotate = conv2tuple(cfg.get('rotate', 0))
        translate_percent = conv2tuple(cfg.get('translate_percent', 0))
        scale = conv2tuple(cfg.get('scale', 1.0))
        return [AffineWithCrop(scale, translate_percent, rotate)]
    elif name == 'segmentation':
        H = cfg['height']
        W = cfg['width']

        def resize_image_to_min(images, random_state, parents, hooks):
            res = []
            for img in images:
                ih, iw = img.shape[:2]

                if ih >= H and iw >= W:
                    res.append(img)
                    continue
                minh = (ih - H) < (iw - W)
                if minh:
                    factor = H / ih
                else:
                    factor = W / iw
                sizes = (factor, factor)
                if isinstance(img, np.ndarray):
                    res.append(imresize_single_image(img, sizes, "linear"))
                else:
                    seg = img.scale(sizes, interpolation='linear')
                    seg.shape = (int(np.round(ih * factor)), int(np.round(iw * factor))) + seg.shape[2:]
                    res.append(seg)
            return res

        return [iaa.Lambda(func_images=resize_image_to_min,
                           func_keypoints=None,
                           func_heatmaps=resize_image_to_min),
                iaa.CropToFixedSize(W, H),
                iaa.Fliplr(p=0.5)]
    elif name == 'randomcrop':
        H = cfg['height']
        W = cfg['width']
        scale = cfg.get('scale')
        if scale is not None:
            return [iaa.Scale({"height": int(H*scale), "width": int(W*scale)}),
                    iaa.size.CropToFixedSize(W, H)]
        else:
            return [iaa.size.CropToFixedSize(W, H)]
    elif name == "cropfrombbox":
        return [CropFromBbox()]
    elif name == "pad_to_fixed_size":
        H = cfg['height']
        W = cfg['width']
        return [iaa.PadToFixedSize(width=W, height=H)]
    elif name == "crop_to_fixed_size":
        H = cfg['height']
        W = cfg['width']
        return [iaa.CropToFixedSize(width=W, height=H)]
    elif name == 'zoom':
        return [iaa.Affine(scale=cfg, name='zoom')]
    elif name in ignore:
        return None
    raise ValueError("Unkown transform {}.".format(name))

