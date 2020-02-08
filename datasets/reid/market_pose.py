from datasets import register_dataset
from datasets.reid_dataset import ReidDataset
from datasets.pose_dataset import PoseDataset
from builders import transform_builder
import pandas as pd
import os
import numpy as np
from datasets.utils import HeaderItem
from datasets.mpii import make_joint_info
from settings import Config


def make_reid_pose_dataset(csv_file, data_dir):
    df = pd.read_csv(csv_file, delimiter=',', header=None)
    data = []
    for idx, datum in df.iterrows():
        d = {}
        joints = np.array(datum[2:]).reshape(16, 2)
        d['coords'] = joints
        d['pid'] = datum[0]
        d['path'] = os.path.join(data_dir, datum[1])
        data.append(d)

    joint_info = make_joint_info()
    header = {}
    header = {'coords': HeaderItem((joint_info.n_joints, 2), np.ndarray((16, 2), dtype=np.float32)),
              'path': HeaderItem((), ""),
              'pid': HeaderItem((), -1)
              }
    info = {
        'joint_info': joint_info,
        'num_joints': joint_info.n_joints
    }
    return data, header, info



@register_dataset("market1501_pose")
class MarketPoseReid(ReidDataset, PoseDataset):
    def __init__(self, *args, **kwargs):
        # this name is relevant for evaluation
        super().__init__("market1501", *args, **kwargs)

    def __getitem__(self, index):
        datum = self.data[index]
        datum = datum.copy()

        img = self.loader_fn(datum['path'])
        if self.transform is not None:
            self.transform.to_deterministic()
            shape = img.shape
            img = self.transform.augment_image(img)
            # the shape of the original image
            datum['coords'] = self.transform.augment_keypoint(datum['coords'], shape)
            # the shape of the augmented image
            datum['coords'] = self.normalize_pose_keypoints(datum['coords'], img.shape)

        datum['img'] = img
        return datum

    @staticmethod
    def build(cfg, *args, **kwargs):
        source_file = cfg['source_file']
        data_dir = Config.MARKET_DATA
        data, header, info = make_reid_pose_dataset(source_file, data_dir)
        transform = transform_builder.build(cfg['transform'], info)
        return MarketPoseReid(data, header, info, transform, *args, **kwargs)
