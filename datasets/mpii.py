"""
Largely adapted from Istvan Sarandi.
Contact: sarandi@vision.rwth-aachen.de

"""
import os
import numpy as np
from utils import cache_result_on_disk
from datasets.utils import loadmat
from datasets.utils import HeaderItem
from datasets.pose_dataset import JointInfo, PoseDataset
from datasets import register_dataset
import socket
from tqdm import tqdm
from metrics import calculate_pckh
import h5py
from builders import transform_builder
from evaluation import Evaluation
from writers.h5 import DynamicH5Writer
from settings import Config


def make_joint_info():
    # Build body joint metadata
    short_names = [
        'r_ank', 'r_kne', 'r_hip', 'l_hip', 'l_kne', 'l_ank', 'pelv', 'thor',
        'neck', 'head', 'r_wri', 'r_elb', 'r_sho', 'l_sho', 'l_elb', 'l_wri']

    full_names = [
        'right ankle', 'right knee', 'right hip', 'left hip', 'left knee',
        'left ankle', 'pelvis', 'thorax', 'neck', 'head', 'right wrist',
        'right elbow', 'right shoulder', 'left shoulder', 'left elbow',
        'left wrist']

    joint_info = JointInfo(short_names, full_names)
    j = joint_info.ids
    joint_info.stick_figure_edges = [
        (j.l_sho, j.l_elb), (j.r_sho, j.r_elb), (j.l_elb, j.l_wri),
        (j.r_elb, j.r_wri), (j.l_hip, j.l_kne), (j.r_hip, j.r_kne),
        (j.l_kne, j.l_ank), (j.r_kne, j.r_ank), (j.neck, j.head),
        (j.pelv, j.thor)]
    return joint_info


# cached_paths are absolute
mpii_cache_path = 'cached/cached_mpii'


@cache_result_on_disk(mpii_cache_path, [0, 2], forced=False)
def make_dataset(mat_path, mpii_root, split):
    joint_info = make_joint_info()
    data = []
    validation_imagelist_path = os.path.join(mpii_root, 'valid_images.txt')
    with open(validation_imagelist_path, 'r') as f:
        validation_filenames = f.read().strip().split('\n')

    s = loadmat(mat_path).RELEASE
    annolist = np.atleast_1d(s.annolist)

    for anno_id, (anno, is_train, rect_ids) in tqdm(enumerate(
            zip(annolist, s.img_train, s.single_person))):
        image_path = os.path.join(mpii_root, 'images', anno.image.name)
        image_filename = os.path.basename(image_path)
        if not os.path.isfile(image_path):
            print("warning file not found! {}.".format(image_path))
            continue

        if is_train and image_filename in validation_filenames:
            anno_belongs_to = "val"
        elif is_train:
            anno_belongs_to = "train"
        else:
            anno_belongs_to = "test"

        if split != anno_belongs_to:
            continue

        rect_ids = np.atleast_1d(rect_ids) - 1
        annorect = np.atleast_1d(anno.annorect)
        for rect_id in rect_ids:
            rect = annorect[rect_id]

            coords = np.full(shape=[joint_info.n_joints, 2], fill_value=np.nan, dtype=np.float32)
            if is_train:
                for joint in np.atleast_1d(rect.annopoints.point):
                    coords[joint.id] = [joint.x, joint.y]

                head_pt1 = np.array([rect.x1, rect.y1], np.float32)
                head_pt2 = np.array([rect.x2, rect.y2], np.float32)
                head_size = np.linalg.norm(head_pt2 - head_pt1) * 0.6
            else:
                # Dummy labels for test set, as these are not publicly available
                head_size = 1
                coords[...] = 0.1

            rough_person_center = np.float32([rect.objpos.x, rect.objpos.y])
            rough_person_size = rect.scale * 200
            # Shift person center down like [Sun et al. arxiv:1711.08229],
            # who say this is common on MPII
            rough_person_center[1] += 0.075 * rough_person_size

            # convert to left, top, right, bottom
            left = rough_person_center[0] - rough_person_size/2
            top = rough_person_center[1] - rough_person_size/2
            right = rough_person_center[0] + rough_person_size/2
            bottom = rough_person_center[1] + rough_person_size/2

            bbox = np.asarray([left, top, right, bottom], dtype=np.float32)
            d = {
                    'anno_id': anno_id,
                    'path': image_path,
                    'coords': coords,
                    'head_size': head_size,
                    'center': rough_person_center,
                    'bbox': bbox,
                }
            data.append(d)

    header = {'coords': HeaderItem((joint_info.n_joints, 2), np.ndarray((16, 2), dtype=np.float32)),
              'head_size': HeaderItem((), -1),
              'path': HeaderItem((), ""),
              'bbox': HeaderItem((4,), np.ndarray((4), dtype=np.float32))}
    info = {
        'joint_info': joint_info,
        'num_joints': joint_info.n_joints
    }
    print(len(data))
    return data, header, info


@register_dataset("mpii")
class Mpii(PoseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("mpii", *args, **kwargs)
       # assert isinstance(self.transform, TransformWithCrop)

    def __getitem__(self, index):
        datum = self.data[index]
        datum = datum.copy()

        img = self.loader_fn(datum['path'])

        # get cropped with keypoints
        x1, y1, x2, y2 = datum['bbox']
        center = datum['center']
        width = x2-x1
        height = y2-y1
        bbox = (width, height)
        self.transform.update([center], [bbox])
        self.transform.to_deterministic()
        shape = img.shape
        img = self.transform.augment_image(img)

        datum['img'] = img

        # the shape of the original image
        datum['coords'] = self.transform.augment_keypoint(datum['coords'], shape)
        # the shape of the augmented image
        datum['coords'] = self.normalize_pose_keypoints(datum['coords'], img.shape)
        return datum

    def __len__(self):
        return len(self.data)


    @staticmethod
    def build(cfg, *args, **kwargs):
        split = cfg['split']
        mat_path = Config.MPII_SOURCE
        data_dir = Config.MPII_DATA
        data, header, info = make_dataset(mat_path, data_dir, split)
        transform = transform_builder.build(cfg['transform'], info)
        dataset = Mpii(data, header, info, transform, *args, **kwargs)
        return dataset

    def get_evaluation(self, model):
        return MpiiEvaluation(model, self)


class MpiiEvaluation(Evaluation):
    """Mpii Evaluation using h5 writer."""
    def __init__(self, model, dataset):
        super().__init__("mpii")
        self.dataset = dataset
        self.joint_info = dataset.info['joint_info']

    def get_writer(self, output_path):
        self.output_file = os.path.join(output_path, 'mpii.h5')


        self.writer = DynamicH5Writer(self.output_file)
        return self.writer

    def before_saving(self, endpoints, data):
        data_to_write = {
            'bbox': data['bbox'].cpu().numpy(),
            'coords': data['coords'].cpu().numpy(),
            'head_size': data['head_size'].cpu().numpy(),
            'pose': endpoints['pose'].cpu().numpy()
        }
        return data_to_write


    def score(self):

        """
        Calculate PCKh according to mpii evaluation procedure.

        embedding_file (string): Path to h5py file that contains
            network output as well as groundtruth.

        """
        with h5py.File(self.output_file, 'r') as f:
            def to_float(array):
                return array.astype(np.float)
            bbox = to_float(np.asarray(f['bbox']))
            left = bbox[:, 0]
            top = bbox[:, 1]
            right = bbox[:, 2]
            bottom = bbox[:, 3]
            width = right - left
            height = bottom - top

            data = np.asarray(f['pose'])

            # test time augmentation not supported
            # data = np.squeeze(data)

            # scale back to original size (has to be same scale as headsize)
            data[:, :, 0] *= width[:, None]
            data[:, :, 1] *= height[:, None]
            # gt data is also transformed
            gt_data = np.asarray(f['coords'])
            # for correct error scale we only scale it back to original size
            gt_data = to_float(gt_data)
            gt_data[:, :, 0] *= width[:, None]
            gt_data[:, :, 1] *= height[:, None]

            head_sizes = np.asarray(f['head_size'])
            head_sizes = to_float(head_sizes)

            ids = self.joint_info.ids

            # exclude pelvis & thorax
            gt_data[:, ids['thor'], :] = np.nan
            gt_data[:, ids['pelv'], :] = np.nan
            def calc_dist(array1, array2):
                return np.linalg.norm(array1 - array2, axis=2)
            dist = calc_dist(gt_data, data)
            pck_all, pck_joint = calculate_pckh(dist, head_sizes)

            score = {}
            # score name string
            sn = "PCKh {} @ {}"
            #threshold: values
            for t, v in pck_joint.items():
                score[sn.format(t, "Head")] = (v[ids['head']] + v[ids['neck']]) / 2
                score[sn.format(t, "Shoulder")] = (v[ids['l_sho']] + v[ids['r_sho']]) / 2
                score[sn.format(t, "Elbow")] = (v[ids['l_elb']] + v[ids['r_elb']]) / 2
                score[sn.format(t, "Wrist")] = (v[ids['l_wri']] + v[ids['r_wri']]) / 2
                score[sn.format(t, "Hip")] = (v[ids['l_hip']] + v[ids['r_hip']]) / 2
                score[sn.format(t, "Knee")] = (v[ids['l_kne']] + v[ids['r_kne']]) / 2
                score[sn.format(t, "Ankle")] = (v[ids['l_ank']] + v[ids['r_ank']]) / 2

            for t, v in pck_all.items():
                score[sn.format(t, "All")] = v

            return score
