import numpy as np
from datasets.dataset import Dataset
from utils import AttrDict


def build_mapping_pairs(ids):
    mirror_pairs = list()
    mapped = set()
    for name, join_id in ids.items():
        if name in mapped:
            continue
        if name.startswith('r_'):
            joint1 = join_id
            name2 = 'l_' + name[2:]
            joint2 = ids[name2]
        elif name.startswith('l_'):
            joint1 = join_id
            name2 = 'r_' + name[2:]
            joint2 = ids[name2]
        else:
            continue
        mirror_pairs.append((joint1, joint2))
        # only adding name 2
        mapped.add(name2)
    return mirror_pairs


class JointInfo:
    """Represents the metadata about the body joints labeled in a particular dataset."""

    def __init__(self, short_names, full_names, stick_figure_edges=None):
        self.ids = AttrDict(dict(zip(short_names, range(len(short_names)))))
        self.short_names = short_names
        self.full_names = full_names
        self.n_joints = len(self.short_names)

        # Joint ID pairs for which an edge should be drawn when visualizing the skeleton
        self.stick_figure_edges = stick_figure_edges

        # The index of the joint on the opposite side (e.g. index of left wrist for index
        # of right wrist). Useful for flip data augmentation
        self.mirror_mapping = list(range(self.n_joints))

        for name in short_names:
            if name[0] == 'r':
                self.mirror_mapping[self.ids[name]] = self.ids['l' + name[1:]]
            elif name[0] == 'l':
                self.mirror_mapping[self.ids[name]] = self.ids['r' + name[1:]]
        self.mirror_mapping_pairs = build_mapping_pairs(self.ids)


class PoseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def normalize_pose_keypoints(self, coords, shape):
        if not self.transform.debug:
            channels, height, width = shape
            coords[:, 0] /= width
            coords[:, 1] /= height
            # set non visible joints to nan
            try:
                dim1, dim2 = np.where(coords < 0)
                coords[dim1] = np.nan
                dim1, dim2 = np.where(coords > 1)
                coords[dim1] = np.nan
            except Exception as e:
                # TODO whats the point of this?
                raise e
        return coords

    def __len__(self):
        return len(self.data)
