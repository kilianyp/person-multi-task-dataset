from datasets import register_dataset
from datasets.reid_dataset import ReidDataset
from datasets.segmentation_dataset import SegmentationDataset
from builders import transform_builder
import os
import numpy as np
from datasets.utils import HeaderItem
from settings import Config
from datasets.utils import make_dataset_default
from datasets.lip import SegInfo, CLASSES as LIP_CLASSES
from transforms.flip_lr_with_pairs import FliplrWithPairs
import imgaug as ia
from PIL import Image


CLASSES = {
    0: "Background",
    1: "Head",
    2: "Lower-body",
    3: "Upper-body",
    4: "Shoes"
}


def make_seg_info():
    id_to_label = CLASSES
    label_to_id = {value: key for key, value in id_to_label.items()}
    class_mapping = {
            "Background": "Background",
            "Hat": "Head",
            "Hair": "Head",
            "Glove": "Upper-body",
            "Sunglasses": "Head",
            "UpperClothes": "Upper-body",
            "Dress": "Upper-body",
            "Coat": "Upper-body",
            "Socks": "Lower-body",
            "Pants": "Lower-body",
            "Jumpsuits": "Upper-body",
            "Scarf": "Upper-body",
            "Skirt": "Lower-body",
            "Face": "Head",
            "Left-arm": "Upper-body",
            "Right-arm": "Upper-body",
            "Left-leg": "Lower-body",
            "Right-leg": "Lower-body",
            "Left-shoe": "Shoes",
            "Right-shoe": "Shoes"
    }

    class_map = np.arange(len(LIP_CLASSES))

    for original_id, c in LIP_CLASSES.items():
        new_class = class_mapping[c]
        new_id = label_to_id[new_class]
        class_map[original_id] = new_id

    return SegInfo(CLASSES, None), class_map


def make_reid_seg_dataset(csv_file, data_dir, seg_dir):
    data, header, info = make_dataset_default(csv_file, data_dir)
    for d in data:
        seg_path = d['path'].split(os.sep)[-2:]
        d['seg_path'] = os.path.join(seg_dir, seg_path[0], seg_path[1] + '.png')

    seg_info, class_mapping = make_seg_info()
    print("seginfo", len(seg_info.id_to_label))
    new_info = {
        'seg_info': seg_info,
        'num_seg_classes': len(seg_info.id_to_label),
        'seg_class_mapping': class_mapping
    }
    info.update(new_info)
    return data, header, info



@register_dataset("market1501_segmentation")
class MarketSegmentationReid(ReidDataset, SegmentationDataset):
    def __init__(self, *args, **kwargs):
        # this name is relevant for evaluation
        super().__init__("market1501", *args, **kwargs)
        seg_info = self.info['seg_info']
        self.seg_mapping = self.info['seg_class_mapping']
        self.flip_prob = kwargs.get('flip_prob', 0.0)
        self.flip_transform = FliplrWithPairs(p=self.flip_prob,
                segmentation_pairs=seg_info.pairs)

    def __getitem__(self, index):
        datum = self.data[index]
        datum = datum.copy()

        img = self.loader_fn(datum['path'])
        seg = np.array(Image.open(datum['seg_path']))
        # convert classes
        seg = self.seg_mapping[seg]

        if self.transform is not None:
            # flip transform is outside the pipeline
            # segmentation label flipping is not yet supported
            # do before possible normalization
            num_seg_classes = self.info['num_seg_classes']

            if self.flip_prob > 0:
                # only execute if the probability is greater 0
                # if the image will be flipped is decided by augmenter
                det_flip = self.flip_transform.to_deterministic()
                img = det_flip.augment_image(img)
                seg = ia.SegmentationMapOnImage(seg, shape=seg.shape, nb_classes=num_seg_classes)
                seg = det_flip.augment_segmentation_maps(seg).get_arr_int()

            self.transform.to_deterministic()
            img = self.transform.augment_image(img)
            seg = self.transform.augment_segmentation(seg, num_seg_classes)

        datum['img'] = img
        # TODO why long?? Otherwise error in loss
        datum['seg'] = np.array(seg, dtype=np.int64)

        return datum

    @staticmethod
    def build(cfg, *args, **kwargs):
        source_file = Config.MARKET_TRAIN
        data_dir = Config.MARKET_DATA
        seg_dir = cfg['seg_dir']
        data, header, info = make_reid_seg_dataset(source_file, data_dir, seg_dir)
        transform = transform_builder.build(cfg['transform'], info)
        return MarketSegmentationReid(data, header, info, transform, *args, **kwargs)
