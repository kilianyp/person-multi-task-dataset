import scipy.io
import os
import csv
import cv2
import scipy.io as io
import h5py
import numpy as np
import warnings
from PIL import Image
from utils import cache_result_on_disk, AttrDict
from collections import namedtuple
import tqdm


HeaderItem = namedtuple('HeaderItem', ['shape', 'default'])


def cv_2_pil_loader(path):
    img = cv2_loader(path)
    return Image.fromarray(img)


def pil_loader(data_path):
    return Image.open(data_path)


def pil_2_cv_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


def cv2_loader(path, mode=cv2.COLOR_BGR2RGB):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, mode) # cv2 is by default BGR
    return image


def make_dataset_default(csv_file, data_dir):
    """Reads in a csv file according to the scheme "target, path".
    """
    header = {"pid": HeaderItem((), -1), "path": HeaderItem((), '')}
    def default_name_fn(row):
        return row['path']

    data, header, dataset_info = make_dataset_unamed(csv_file, data_dir, default_name_fn, header)
    dataset_info['mean'] = [0.485, 0.456, 0.406] # default values for imgnet
    dataset_info['std'] = [0.229, 0.224, 0.225]
    return data, header, dataset_info


def make_dataset_mot_old(csv_file, data_dir, limit):
    """Reads in a csv file according to the scheme of mot files.
    Args:
        limit: Number of images that are read in.
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for id, row in enumerate(reader):
            if limit is not None and id >= limit:
                break
            target = row[1]
            file_name = row[-1]
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir)
                continue
            imgs.append([file_dir, target])
    return imgs


def mot_name_fn(row):
    return "{:06}.jpg".format(int(row["frame"]))


def make_dataset_named_mot(csv_file, data_dir):
    return make_dataset_named(csv_file, data_dir, mot_name_fn)


def parse(row, header):
    from collections import OrderedDict
    dic = OrderedDict()
    for idx, (col, dtype) in enumerate(header.items()):
        dic[col] = row[idx]
    return dic


def make_dataset_mot(csv_file, data_dir):

    header = ["frame", "pid", "left", "top",
            "width", "height", "confidence"]
    return make_dataset_unamed(csv_file, data_dir, mot_name_fn, header)


def make_dataset_unamed(csv_file, data_dir, image_name_fn, header):
    """Reads in a csv file according to the scheme of mot files.
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            row = parse(row, header)
            file_name = image_name_fn(row)
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir)
                continue
            data = {key: row[key] for key, default in header.items()}
            data['pid'] = int(data['pid'])
            data['path'] = file_dir
            data['row_idx'] = idx
            # list so that targets can be rewritten later
            imgs.append(data)
    return imgs, header, {}


def make_dataset_named(csv_file, data_dir, image_name_fn):
    """Reads in a csv file with named columns.
    Args:
        csv_file, data_dir where the image is stored
    """
    imgs = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for id, row in enumerate(reader):
            file_name = image_name_fn(row)
            file_dir = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_dir):
                warnings.warn("File %s could not be found and is skipped!" % file_dir) 
                continue
            # list so that targets can be rewritten later
            row['path'] = file_dir
            imgs.append(row)
    # TODO needs default values
    header = reader.fieldnames
    return imgs, header, {}


def read_from_mat(path):
    """Reads from a Matlab file.
    Returns (dic): A dictionary with the different variables in keys."""
    try:
        """Used by MATLAB before.
        """
        return io.loadmat(path)
    except NotImplementedError:
        """Used by MATLAB 7.3"""
        arrays = {}
        with h5py.File(path, 'r') as f:
            for k, v in f.items():
                print("Reading in: {}".format(k))
                arrays[k] = np.array(v)
        return arrays


def loadmat(path):
    """This function is better than scipy.io.loadmat as it cures the problem of not properly
    recovering Python dictionaries from mat files. It transforms all entries which are still
    mat-objects.

    Adapted from Istvan Sarandi.
    Contact: sarandi@vision.rwth-aachen.de
    """

    dic = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)

    def _cure(elem):
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            return _to_attrdict(elem)
        elif isinstance(elem, np.ndarray) and elem.ndim == 1:
            return _to_list(elem)
        else:
            return elem

    def _to_attrdict(mat_struct):
        return AttrDict(
            {field_name: _cure(getattr(mat_struct, field_name))
             for field_name in mat_struct._fieldnames})

    def _to_list(ndarray):
        return [_cure(elem) for elem in ndarray]

    return AttrDict({k: _cure(v) for k, v in dic.items()})


def filter_junk(data):
    new_data = []
    for d in data:
        if d['pid'] <= 0:
            continue

        new_data.append(d)

    return new_data


@cache_result_on_disk('./cached/mean', forced=False)
def compute_mean(data, loader):
    mean = np.zeros(3)
    std = np.zeros(3)
    for d in tqdm(data):
        img = loader(d['path'])
        img = np.array(img).reshape(-1, 3)
        mean += img.mean(axis=0)
        std += img.std(axis=0)

    mean /= len(data)
    std /= len(data)
    print('    Mean: %.4f, %.4f, %.4f' % (mean[0], mean[1], mean[2]))
    print('    Std:  %.4f, %.4f, %.4f' % (std[0], std[1], std[2]))

    return mean, std

