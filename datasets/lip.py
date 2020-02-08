"""
There are multiple folders. One for LIP (What we want), one for Fashion Design (Ac), and one for multiple people (CIHP)


Folder Structure
Testing_images/Testing_images/testing_images: Test images
TrainVal_images/TrainVal_images/train_images: train images, ignore text files
TrainVal_images/TrainVal_images/val_images: train images, ignore text files
TrainVal_parsing_annotations/TrainVal_images/train_images: Train segmetation map
TrainVal_parsing_annotations/TrainVal_images/val_images: Val segmentation map
TrainVal_pose_annotations: json files of pose annotation

from source with caching.

"""
import pandas as pd
from logger import get_logger
import os
from .pose_dataset import JointInfo
from datasets import register_dataset
from datasets.utils import HeaderItem
from datasets.pose_dataset import PoseDataset
from builders import transform_builder
import numpy as np
from settings import Config
from evaluation import Evaluation
import torch
from writers.dummy import DummyWriter
from writers.memory import MemoryWriter
from utils import cache_result_on_disk
from metrics import calculate_pckh
from metrics import calc_seg_score
from transforms.flip_lr_with_pairs import FliplrWithPairs
import imgaug as ia
from metrics import fast_hist
import cv2
from utils import save_seg_as_png

COLORMAP = np.array(
    [
        [0, 0, 0], #black
        [101.,  55.,   0.],
        [110., 117.,  14.],
        [  6., 194., 172.],
        [199., 159., 239.],
        [203.,  65., 107.],
        [ 56.,   2., 130.],
        [206., 179.,   1.],
        [224.,  63., 216.],
        [  4., 217., 255.],
        [ 17., 135.,  93.],
        [251., 125.,   7.],
        [255.,   4., 144.],
        [137., 254.,   5.],
        [252.        , 186.67843137, 160.63137255], # Red
        [197.68627451, 218.86666667, 238.89019608], # Blue
        [250.81176471, 105.2627451 ,  73.52941176], # Red
        [106.35686275, 173.56078431, 213.74901961], # Blue
        [202.10588235,  23.78823529,  28.81176471], # Red
        [197.68627451, 218.86666667, 238.89019608], # Blue
    ], dtype=np.uint8
)

def make_joint_info():
    short_names = [
        'r_ank', 'r_kne', 'r_hip', 'l_hip', 'l_kne', 'l_ank', 'b_pelv', 'b_spine',
        'b_neck', 'b_head', 'r_wri', 'r_elb', 'r_sho', 'l_sho', 'l_elb', 'l_wri']

    full_names = [
        'right ankle', 'right knee', 'right hip', 'left hip', 'left knee',
        'left ankle', 'pelvis', 'spine', 'neck', 'head', 'right wrist',
        'right elbow', 'right shoulder', 'left shoulder', 'left elbow',
        'left wrist']

    joint_info = JointInfo(short_names, full_names)
    j = joint_info.ids
    joint_info.stick_figure_edges = [
        (j.l_sho, j.l_elb), (j.r_sho, j.r_elb), (j.l_elb, j.l_wri),
        (j.r_elb, j.r_wri), (j.l_hip, j.l_kne), (j.r_hip, j.r_kne),
        (j.l_kne, j.l_ank), (j.r_kne, j.r_ank), (j.b_neck, j.b_head),
        (j.b_pelv, j.b_spine)]
    return joint_info

CLASSES = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Glove",
    4: "Sunglasses",
    5: "UpperClothes",
    6: "Dress",
    7: "Coat",
    8: "Socks",
    9: "Pants",
    10: "Jumpsuits",
    11: "Scarf",
    12: "Skirt",
    13: "Face",
    14: "Left-arm",
    15: "Right-arm",
    16: "Left-leg",
    17: "Right-leg",
    18: "Left-shoe",
    19: "Right-shoe"
}


# input file
COLS = ["image_id",
        "r_ank_x", "r_ank_y", "r_ank_v",
        "r_kne_x", "r_kne_y", "r_kne_v",
        "r_hip_x", "r_hip_y", "r_hip_v",
        "l_hip_x", "l_hip_y", "l_hip_v",
        "l_kne_x", "l_kne_y", "l_kne_v",
        "l_ank_x", "l_ank_y", "l_ank_v",
        "b_pel_x", "b_pel_y", "b_pel_v",
        "b_spi_x", "b_spi_y", "b_spi_v",
        "b_nec_x", "b_nec_y", "b_nec_v",
        "b_hea_x", "b_hea_y", "b_hea_v",
        "r_wri_x", "r_wri_y", "r_wri_v",
        "r_elb_x", "r_elb_y", "r_elb_v",
        "r_sho_x", "r_sho_y", "r_sho_v",
        "l_sho_x", "l_sho_y", "l_sho_v",
        "l_elb_x", "l_elb_y", "l_elb_v",
        "l_wri_x", "l_wri_y", "l_wri_v"]


class SegInfo(object):
    # pickle does not like namedtuple
    def __init__(self, id_to_label, pairs):
        self.id_to_label = id_to_label
        self.pairs = pairs

def make_seg_info():
    id_to_label = CLASSES
    label_to_id = {value: key for key, value in id_to_label.items()}
    def build_pairs(label_to_id):
        pairs = dict()
        for label in label_to_id:
            if label.startswith('Left'):
                pair1 = label_to_id[label]
                label2 = 'Right' + label[len('Left'):]
                pair2 = label_to_id[label2]
            elif label.startswith('Right'):
                pair1 = label_to_id[label]
                label2 = 'Left' + label[len('Right'):]
                pair2 = label_to_id[label2]
            else:
                continue
            pairs[pair1] = pair2
        return pairs

    pairs = build_pairs(label_to_id)
    return SegInfo(id_to_label, pairs)


#@cache_result_on_disk('cached/lip', [0, 1, 2], forced=False)
def make_dataset(data_path, split="train", mapping=""):
    """
    Makes the LIP dataset.
    TODO Test set will not work.
    """
    # load images
    logger = get_logger()

    if split == "train":
        img_data_path = os.path.join(data_path, 'train_images')
        seg_data_path = os.path.join(data_path, 'TrainVal_parsing_annotations', 'train_segmentations')
        pose_anno_path = os.path.join(data_path, 'TrainVal_pose_annotations', 'lip_train_set.csv')
    elif split == "val":
        img_data_path = os.path.join(data_path, 'val_images')
        seg_data_path = os.path.join(data_path, 'TrainVal_parsing_annotations', 'val_segmentations')
        pose_anno_path = os.path.join(data_path, 'TrainVal_pose_annotations', 'lip_val_set.csv')
    elif split == "test":
        # TODO
        img_data_path = os.path.join(data_path, 'testing_images')
        seg_data_path = None
        pose_anno_path = None
    else:
        raise ValueError("Unknown split in LIP")

    joint_info = make_joint_info()
    data = []
    if pose_anno_path is not None:
        # make image name index
        pose_anno_df = pd.read_csv(pose_anno_path, header=None, names=COLS, index_col=0)

    image_names = os.listdir(img_data_path)
    for image_name in image_names:
        image_id = image_name[:-len('.jpg')]
        img_path = os.path.join(img_data_path, image_name)
        if not os.path.isfile(img_path):
            logger.warning('File %s was not found', img_path)
            continue

        d = {
            'path': img_path
        }

        if seg_data_path is not None:
            seg_path = os.path.join(seg_data_path, image_id + '.png')

            if not os.path.isfile(seg_path):
                logger.warning('File %s was not found', seg_path)
                continue
            d['seg_path'] = seg_path

        if pose_anno_path is not None:
            coords = pose_anno_df.loc[image_name]
            coords = coords.reshape(-1, 3)
            # drop visual column
            coords = coords[:, [0, 1]]
            head_size = None

            # TODO  Is this correct
            head_size = np.linalg.norm(coords[joint_info.ids.b_head] - coords[joint_info.ids.b_neck])
            d['coords'] = coords
            d['head_size'] = head_size
        data.append(d)


    header = {
                'path': HeaderItem((), ""),
                'coords': HeaderItem((), ""),
                # seg is for the actually loaded data TODO
                'seg': HeaderItem((), "")
             }
    seg_info = make_seg_info()
    info = {
        'joint_info': joint_info,
        'num_joints': joint_info.n_joints,
        'seg_info': seg_info,
        'num_seg_classes': len(CLASSES),
        'split': split
    }
    if not mapping == "":
        if mapping == "market1501":
            from datasets.reid.market_seg import make_seg_info as make_market_seg_info
            info['seg_info'], info['seg_mapping'] = make_market_seg_info()
            info['num_seg_classes'] = len(info['seg_info'].id_to_label)
        else:
            raise ValueError(mapping)
    else:
        info['seg_mapping'] = None
    logger.info("num_seg_classes %d", info['num_seg_classes'])
    return data, header, info


from datasets.image_dataset import FullImageDataset

@register_dataset('lip_test')
class LipTest(FullImageDataset):
    @staticmethod
    def build(cfg, *args, **kwargs):
        data_dir = Config.LIP_DATA
        data, header, info = make_dataset(data_dir, "test")
        transform = transform_builder.build(cfg['transform'], info)
        dataset = LipTest("LIP", data, header, info, transform, *args, **kwargs)
        return dataset


@register_dataset('lip')
class Lip(PoseDataset):
    """
    Look into person
    """
    def __init__(self, data, header, info, flip_prob, *args, **kwargs):
        super().__init__("lip", data, header, info, *args, **kwargs)
        seg_info = info['seg_info']
        joint_info = info['joint_info']
        self.flip_prob = flip_prob
        self.flip_transform = FliplrWithPairs(p=flip_prob,
                keypoint_pairs=joint_info.mirror_mapping_pairs,
                segmentation_pairs=seg_info.pairs)
        self.logger = get_logger()
        self.train = info['split'] == 'train'

    def __getitem__(self, index):
        datum = self.data[index]
        datum = datum.copy()

        img = self.loader_fn(datum['path'])
        shape = img.shape
        coords = datum['coords']
        # image is a 3 channel png with identical channels
        seg = np.array(self.loader_fn(datum['seg_path']))[:, :, 0]

        if not self.info['seg_mapping'] is None:
            seg = self.info['seg_mapping'][seg]

        if self.transform is not None:
            # flip transform is outside the pipeline
            # segmentation label flipping is not yet supported
            # do before possible normalization
            num_seg_classes = self.info['num_seg_classes']

            if self.flip_prob > 0:
                # only execute if the probability is greater 0
                # if the image will be flipped is decided by augmenter
                det_flip = self.flip_transform.to_deterministic()
                #det_flip = self.flip_transform
                img = det_flip.augment_image(img)
                seg = ia.SegmentationMapOnImage(seg, shape=seg.shape, nb_classes=num_seg_classes)
                seg = det_flip.augment_segmentation_maps(seg).get_arr_int()

                keypoints_on_image = ia.KeypointsOnImage.from_coords_array(coords, shape=shape)
                keypoints_on_image = det_flip.augment_keypoints([keypoints_on_image])
                coords = keypoints_on_image[0].get_coords_array()

            self.transform.to_deterministic()
            img = self.transform.augment_image(img)

            # apply other transformations only for training
            if self.train:
                seg = self.transform.augment_segmentation(seg, num_seg_classes)
            # the shape of the original image
            coords = self.transform.augment_keypoint(coords, shape)
            # the shape of the augmented image
            coords = self.normalize_pose_keypoints(coords, img.shape)

        # we need to save the shape to restore the orginal coordinates
        datum['height'] = shape[0]
        datum['width'] = shape[1]
        datum['coords'] = coords
        datum['img'] = img
        # TODO why long?? Otherwise error in loss
        datum['seg'] = np.array(seg, dtype=np.int64)

        return datum

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build(cfg, *args, **kwargs):
        split = cfg['split']
        evaluate = cfg.get('evaluate', 'both')
        #default to zero to avoid messing up validation
        flip_prob = cfg.get('flip_prob', 0.0)
        data_dir = Config.LIP_DATA
        mapping = cfg.get('mapping', "")
        data, header, info = make_dataset(data_dir, split, mapping)
        transform = transform_builder.build(cfg['transform'], info)
        dataset = Lip(data, header, info, flip_prob, transform, *args, **kwargs)
        # TODO very temporay solution
        # Looking for a better solution building the evaluation
        # to avoid passing too many parameters.
        dataset.evaluate_mode = evaluate
        if evaluate == "pose":
            dataset.train = True
        return dataset

    def get_evaluation(self, model):
        pose = segmentation = False

        # We cannot just check the model output
        # because pose and segmentation need different batch sizes
        if self.evaluate_mode == 'pose':
            pose = True
        elif self.evaluate_mode == 'segmentation':
            segmentation = True
        else:
            pose = segmentation = True

        endpoints = model.module.endpoints
        if 'pose' not in endpoints:
            if pose:
                self.logger.warning("Model does not have pose output.")
                pose = False
        if 'sem-logits' not in endpoints:
            if segmentation:
                self.logger.warning("Model does not have segmentation output.")
                segmentation = False

        joint_info = self.info['joint_info']
        num_seg_classes = self.info['num_seg_classes']
        seg_mapping = self.info['seg_mapping']
        id_to_label = self.info['seg_info'].id_to_label
        if pose and segmentation:
            print("LIP: Pose and Segmentation Evaluation started")
            return LipPoseSegmentationEvaluation(model, joint_info, num_seg_classes, seg_mapping, id_to_label)
        elif pose:
            print("LIP: Pose Evaluation started")
            joint_info = self.info['joint_info']
            return LipPoseEvaluation(model, joint_info)
        elif segmentation:
            print("LIP: Segmentation Evaluation started")
            return LipSegmentationEvaluation(model, num_seg_classes, seg_mapping, id_to_label)

        raise RuntimeError("Not the expected outputs available")

    def get_test(self, model):
        raise NotImplementedError


class LipSegmentationTest(object):
    def __init__(self, model):
        self.model = model
        self.path = "test/lip"

    def write(self, data, base_path):
        endpoints = self.model.module.infere(data)
        predictions = torch.argmax(endpoints['sem-logits'], dim=1).detach().cpu().numpy()
        # batch size of one
        gts = data['seg'].detach().cpu().numpy()
        predictions = predictions.astype(np.uint8)
        for pred, gt in zip(predictions, gts):
            # mapping needs to be per evaluation, map gt too if
            # necessary
            if not self.seg_mapping is None:
                pred = self.seg_mapping[pred].astype(np.uint8)
            # we disallowed changing the gt map in the dataset
            if pred.shape != gt.shape:
                # nearest neighbour is the only useful interpolation
                pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
            path = os.path.join(self.path, data['path'])
            save_seg_as_png(path, pred)



class LipSegmentationEvaluation(Evaluation):
    def __init__(self, model, num_classes, seg_mapping, id_to_label):
        super().__init__("Lip")
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        # the mapping depends on the model and the dataset it
        # was trained on and which mapping was used there
        # do we want to evaluate on full or not?
        # model output size is not 20, dont need mapping
        # model output size is 20, need mapping

        # TODO what about trained on one mapping, evaluated on another mapping
        if seg_mapping is None:
            # don't evaluate mapped
            self.seg_mapping = None
            if not model.module.seg_mapping is None:
                print("Warning model was mapped to different classes. Using this mapping.")
                self.seg_mapping = model.module.seg_mapping
        else:
            # evaluate mapped
            # check model if model output is already mapped
            if model.module.seg_mapping is None:
                # has not been mapped
                self.seg_mapping = seg_mapping
                print("mapping output to", seg_mapping)
            else:
                # has alread been mapped, we don't need to change the output
                self.seg_mapping = None
                print("output is already mapped.")
            self.name = 'mapped'

        self.id_to_label = id_to_label

    def get_writer(self, output_path):
        # for now do everything in memory
        self.writer = DummyWriter()
        return self.writer

    def before_saving(self, endpoints, data):
        # Change to Update and remove get_writer function?
        predictions = torch.argmax(endpoints['sem-logits'], dim=1).detach().cpu().numpy()
        # batch size of one
        gts = data['seg'].detach().cpu().numpy()
        predictions = predictions.astype(np.uint8)
        for pred, gt in zip(predictions, gts):
            # mapping needs to be per evaluation, map gt too if
            # necessary
            if not self.seg_mapping is None:
                pred = self.seg_mapping[pred].astype(np.uint8)
            # we disallowed changing the gt map in the dataset
            if pred.shape != gt.shape:
                # nearest neighbour is the only useful interpolation
                pred = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

            self.hist += fast_hist(gt.flatten(), pred.flatten(), self.num_classes)

        return {}

    def score(self):
        score = calc_seg_score(self.hist, self.id_to_label)
        return score

class LipPoseEvaluation(Evaluation):
    def __init__(self, model, joint_info):
        super().__init__("Lip")
        self.joint_info = joint_info

    def get_writer(self, output_path):
        # for now do everything in memory
        self.writer = MemoryWriter()
        return self.writer

    def before_saving(self, endpoints, data):
        data_to_write = {
            "pose": endpoints['pose'].cpu(),
            "coords": data['coords'],
            "head_size": data['head_size'],
            "height": data['height'],
            "width": data['width']
        }
        return data_to_write

    @staticmethod
    def _score(pose, coords, height, width, head_size, ids):
        # no inplace
        pose = pose.copy()
        coords = coords.copy()

        # coords are between 0 and 1, rescale for correct error
        # broadcast to all joints

        pose[:, :, 0] *= width[:, None]
        pose[:, :, 1] *= height[:, None]

        coords[:, :, 0] *= width[:, None]
        coords[:, :, 1] *= height[:, None]
        def calc_dist(array1, array2):
            return np.linalg.norm(array1 - array2, axis=2)
        # TODO ignore head not visible in evaluation
        dist = calc_dist(pose, coords)
        pck_all, pck_joint = calculate_pckh(dist, head_size)

        score = {}
        sn = "PCKh {} @ {}"
        #threshold: values
        for t, v in pck_joint.items():
            score[sn.format(t, "Head")] = (v[ids['b_head']] + v[ids['b_neck']]) / 2
            score[sn.format(t, "Shoulder")] = (v[ids['l_sho']] + v[ids['r_sho']]) / 2
            score[sn.format(t, "Elbow")] = (v[ids['l_elb']] + v[ids['r_elb']]) / 2
            score[sn.format(t, "Wrist")] = (v[ids['l_wri']] + v[ids['r_wri']]) / 2
            score[sn.format(t, "Hip")] = (v[ids['l_hip']] + v[ids['r_hip']]) / 2
            score[sn.format(t, "Knee")] = (v[ids['l_kne']] + v[ids['r_kne']]) / 2
            score[sn.format(t, "Ankle")] = (v[ids['l_ank']] + v[ids['r_ank']]) / 2

        for t, v in pck_all.items():
            score[sn.format(t, "All")] = v
        return score

    def score(self):
        data = self.writer.data
        height = np.concatenate(data['height'])
        width = np.concatenate(data['width'])
        head_size = np.concatenate(data['head_size'])
        pose = np.concatenate(data['pose']) # prediction
        coords = np.concatenate(data['coords']) # gt
        return self._score(pose, coords, height, width, head_size, self.joint_info.ids)


class LipPoseSegmentationEvaluation(Evaluation):
    def __init__(self, model, joint_info, num_seg_classes, seg_mapping, id_to_label):
        super().__init__("Lip")
        self.pose = LipPoseEvaluation(model, joint_info)
        self.seg = LipSegmentationEvaluation(model, num_seg_classes, seg_mapping, id_to_label)

    def get_writer(self, output_path):
        self.writer = MemoryWriter()
        self.seg.writer = self.writer
        self.pose.writer = self.writer
        return self.writer

    def before_saving(self, endpoints, data):
        pose_data = self.pose.before_saving(endpoints, data)
        seg_data = self.seg.before_saving(endpoints, data)
        return {**pose_data, **seg_data}

    def score(self):
        pose_score = self.pose.score()
        seg_score = self.seg.score()
        return {**pose_score, **seg_score}
