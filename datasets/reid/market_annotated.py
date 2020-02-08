from datasets import register_dataset
from datasets.reid_dataset import ReidDataset
from datasets.pose_dataset import PoseDataset
from builders import transform_builder
import os
import numpy as np
from datasets.utils import HeaderItem
from datasets.lip import make_joint_info
from datasets.reid.market_seg import make_seg_info
from settings import Config
from PIL import Image
import csv
from datasets.reid_dataset import make_pid_dataset


def make_reid_annotated(csv_file, data_dir, seg_dir=None, attribute_file=None, pose_file=None, pid_limit=None):
    data, header, info = make_pid_dataset(csv_file, data_dir, pid_limit)
    if pose_file is not None:
        joint_info = make_joint_info()
        info['joint_info'] = joint_info
        info['num_joints'] = joint_info.n_joints
        header['coords'] = HeaderItem((joint_info.n_joints, 2), np.ndarray((16, 2), dtype=np.float32)),
        with open(pose_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            length = sum(1 for row in reader)
            f.seek(0)
            if length == len(data):
                for d, row in zip(data, reader):
                    row = list(map(float, row))
                    coords = np.asarray(row).reshape(16, 2)
                    d['coords'] = coords
                    d['head_size'] = np.linalg.norm(coords[joint_info.ids.b_head] - coords[joint_info.ids.b_neck])
            else:
                data_iter = iter(data)
                d = next(data_iter)
                for idx, row in enumerate(reader):
                    if idx == d['row_idx']:
                        row = list(map(float, row))
                        coords = np.asarray(row).reshape(16, 2)
                        d['coords'] = coords
                        d['head_size'] = np.linalg.norm(coords[joint_info.ids.b_head] - coords[joint_info.ids.b_neck])
                        try:
                            d = next(data_iter)
                        except StopIteration:
                            break


    if seg_dir is not None:
        seg_info, class_mapping = make_seg_info()
        print("seginfo", len(seg_info.id_to_label))
        info['seg_info'] = seg_info
        info['num_seg_classes'] = len(seg_info.id_to_label)
        info['seg_class_mapping'] = class_mapping
        for d in data:
            seg_path = d['path'].split(os.sep)[-2:]
            d['seg_path'] = os.path.join(seg_dir, seg_path[0], seg_path[1] + '.png')
            if not os.path.isfile(d['seg_path']):
                raise RuntimeError("File not found {}".format(d['seg_path']))

    if attribute_file is not None:
        with open(attribute_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            attributes = reader.fieldnames
            for attribute in attributes:
                header[attribute] = HeaderItem((1,), 0)
            # TODO this should be the same as in pose
            # it requires that the dataset is sorted by PID
            for d, row in zip(data, reader):
                for key, value in row.items():
                    d[key] = int(value)

    return data, header, info


@register_dataset("market1501_annotated")
class MarketAnnotated(ReidDataset, PoseDataset):
    def __init__(self, annotations, *args, **kwargs):
        # this name is relevant for evaluation
        super().__init__("market1501", *args, **kwargs)
        get_item_fns = []
        self.annotations = annotations
        for annotation in annotations:
            if annotation == "pose":
                get_item_fns.append(self.get_pose)
            if annotation == "segmentation":
                self.seg_mapping = self.info['seg_class_mapping']
                get_item_fns.append(self.get_segmentation)
        self.get_item_fns = get_item_fns

    def get_segmentation(self, datum):
        seg = np.array(Image.open(datum['seg_path']))
        # convert classes
        seg = self.seg_mapping[seg]
        num_seg_classes = self.info['num_seg_classes']
        seg = self.transform.augment_segmentation(seg, num_seg_classes)
        datum['seg'] = np.array(seg, dtype=np.int64)

    def get_pose(self, datum):
        shape = datum['shape']
        datum['coords'] = self.transform.augment_keypoint(datum['coords'], shape)
        # the shape of the augmented image
        shape_new = datum['img'].shape
        datum['coords'] = self.normalize_pose_keypoints(datum['coords'], shape_new)

    def __getitem__(self, index):
        datum = self.data[index]
        datum = datum.copy()
        img = self.loader_fn(datum['path'])
        self.transform.to_deterministic()
        datum['shape'] = img.shape

        img = self.transform.augment_image(img)
        datum['img'] = img

        for get_item_fn in self.get_item_fns:
            get_item_fn(datum)

        return datum

    @staticmethod
    def build(cfg, *args, **kwargs):
        source_file = Config.MARKET_TRAIN
        data_dir = Config.MARKET_DATA
        attribute_file = cfg.get('attribute_file')
        pose_file = cfg.get('pose_file')
        seg_dir = cfg.get('seg_dir')
        annotations = []
        if not pose_file is None:
            annotations.append("pose")
        if not seg_dir is None:
            annotations.append("segmentation")

        pid_limit = cfg.get('pid_limit')

        data, header, info = make_reid_annotated(source_file, data_dir,
                seg_dir=seg_dir,
                attribute_file=attribute_file,
                pose_file=pose_file,
                pid_limit=pid_limit)
        transform = transform_builder.build(cfg['transform'], info)
        return MarketAnnotated(annotations, data, header, info, transform, *args, **kwargs)


    def get_evaluation(self, model):
        raise RuntimeError("use test dataset")


@register_dataset("market1501_annotated_test")
class MarketAnnotatedTest(MarketAnnotated):
    def __init__(self, annotations, data, header, info, transform, *args, **kwargs):
        super().__init__(annotations, data, header, info, transform, *args, **kwargs)

    @staticmethod
    def build(cfg, *args, **kwargs):
        source_file = Config.MARKET_TEST
        data_dir = Config.MARKET_DATA
        pose_file = cfg.get('pose_file')
        seg_dir = cfg.get('seg_dir')
        annotations = []
        attribute_file = None

        if not pose_file is None:
            annotations.append("pose")
        if not seg_dir is None:
            annotations.append("segmentation")

        data, header, info = make_reid_annotated(source_file, data_dir,
                seg_dir=seg_dir,
                attribute_file=attribute_file,
                pose_file=pose_file)
        transform = transform_builder.build(cfg['transform'], info)
        return MarketAnnotatedTest(annotations, data, header, info, transform, *args, **kwargs)

    def get_evaluation(self, model):
        return MarketAnnotatedEvaluation(model, self.annotations, self.info)


from evaluation import Evaluation
from writers.h5 import DynamicH5Writer
from datasets.lip import LipPoseEvaluation
import h5py
import torch
from metrics import fast_hist
from metrics import calc_seg_score
class MarketAnnotatedEvaluation(Evaluation):
    def __init__(self, model, annotations, dataset_info):
        self.name = "Market1501"
        endpoints = model.module.endpoints
        self.pose = False
        if "pose" in endpoints and "pose" in annotations:
            self.joint_info = dataset_info['joint_info']
            self.pose = True
            print("Evaluating pose")

        self.segmentation = False
        if "sem-logits" in endpoints and "segmentation" in annotations:
            self.seg_info = dataset_info['seg_info']
            num_seg_classes = dataset_info['num_seg_classes']
            self.id_to_label = self.seg_info.id_to_label
            self.num_seg_classes = num_seg_classes
            self.hist = np.zeros((num_seg_classes, num_seg_classes))
            self.segmentation = True
            print("Evaluating segmentation")
            # TODO need to check if the model was trained on more classes
            # model from, dataset to, have some logic that creates this mapping
            if model.module.trained_on['num_seg_classes'] != num_seg_classes:
                self.seg_mapping = dataset_info['seg_class_mapping']
            else:
                self.seg_mapping = None

    def get_writer(self, output_path):
        self.output_path = output_path
        self.output_file = os.path.join(output_path, self.name + '.h5')
        self.writer = DynamicH5Writer(self.output_file)
        return self.writer

    def before_saving(self, endpoints, data):
        data_to_write = {}

        if self.pose:
            data_to_write['coords'] = data['coords'].cpu().numpy()
            data_to_write['pose'] = endpoints['pose'].cpu().numpy()
            data_to_write['head_size'] = data['head_size'].cpu().numpy()

        if self.segmentation:
            predictions = torch.argmax(endpoints['sem-logits'], dim=1).detach().cpu().numpy()
            gts = data['seg'].detach().cpu().numpy()
            for gt, pred in zip(gts, predictions):
                if not self.seg_mapping is None:
                    pred = self.seg_mapping[pred]
                self.hist += fast_hist(gt.flatten(), pred.flatten(), self.num_seg_classes)
        return data_to_write

    def score(self):
        scores = {}
        if self.segmentation:
            score = calc_seg_score(self.hist, self.id_to_label)
            scores.update(score)
        if self.pose:
            with h5py.File(self.output_file, 'r') as f:
                height = np.asarray([128])
                width = np.asarray([64])
                head_size = np.asarray(f['head_size'])
                pose = np.asarray(f['pose']) # prediction
                coords = np.asarray(f['coords']) # gt
                score = LipPoseEvaluation._score(pose, coords, height, width, head_size, self.joint_info.ids)
                scores.update(score)

        return scores
