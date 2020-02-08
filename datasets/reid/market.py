from datasets.reid_dataset import ReidDataset, ReidTestDataset, ReidEvaluation
from datasets.reid_dataset import make_pid_dataset
from datasets.utils import make_dataset_default
from datasets import register_dataset
from builders import transform_builder
from settings import Config
from metrics import calc_euclidean
import os
import re
import numpy as np


@register_dataset("market1501")
class MarketReid(ReidDataset):
    def __init__(self, *args, **kwargs):
        super().__init__("market1501", *args, **kwargs)

    def __getitem__(self, index):
        data = self.data[index]
        copied = data.copy()
        img = self.loader_fn(copied['path'])
        if self.transform is not None:
            self.transform.to_deterministic()
            img = self.transform.augment_image(img)

        copied['img'] = img
        return copied

    @staticmethod
    def build(cfg, *args, **kwargs):
        source_file = Config.MARKET_TRAIN
        data_dir = Config.MARKET_DATA
        limit = cfg.get('pid_limit')
        data, header, info = make_pid_dataset(source_file, data_dir, limit)
        transform = transform_builder.build(cfg['transform'], info)
        return MarketReid(data, header, info, transform, *args, **kwargs)


@register_dataset("market1501_test")
class MarketReidTest(ReidTestDataset):
    def __init__(self, test_data, query_data, metric, *args, **kwargs):
        super().__init__("market1501", metric, test_data, query_data, *args, **kwargs)

    @staticmethod
    def build(cfg, *args, **kwargs):
        test_source = Config.MARKET_TEST
        query_source = Config.MARKET_QUERY
        data_dir = Config.MARKET_DATA
        metric = cfg['metric']

        # todo different augmentations for test and query?
        test_data, header, info = make_dataset_default(test_source, data_dir)
        query_data, _, _ = make_dataset_default(query_source, data_dir)
        transform = transform_builder.build(cfg['transform'], info)

        return MarketReidTest(test_data, query_data, metric, header, info, transform, *args, **kwargs)

    def get_evaluation(self, model):
        return ReidEvaluation(self, calc_euclidean, Matcher)


class Matcher(object):
    """
    https://github.com/VisualComputingInstitute/triplet-reid/blob/master/excluders/market1501.py
    """
    def __init__(self, gallery_fids):
        # Setup a regexp for extracing the required info.
        self.regexp = re.compile('(\S+)_c(\d+)s(\d+)_.*')

        # Parse the gallery_set
        self.gallery_pids, self.gallery_cids = self.parse(gallery_fids)

    def parse(self, fids):
        pids = []
        cids = []
        for fid in fids:
            # Extract the basename
            basename = os.path.basename(fid)
            filename = os.path.splitext(basename)[0]

            # Extract the pid and cid
            pid, cid, _ = self.regexp.match(filename).groups()
            pids.append(pid)
            cids.append(cid)

        return np.asarray(pids), np.asarray(cids)

    def __call__(self, query_fids):
        # By default the Market-1501 evaluation ignores same pids in one camera
        # and the distractor images (pid=-1).
        pids, cids = self.parse(np.atleast_1d(query_fids))

        # Ignore same pid image within the same camera
        camera_matches = self.gallery_cids[None] == cids[:,None]
        pid_matches = self.gallery_pids[None] == pids[:,None]
        mask = np.logical_and(camera_matches, pid_matches)

        # Remove all with the -1 pid.
        junk_images = np.repeat(self.gallery_pids[None], len(pids), 0) == '-1'
        mask = np.logical_or(mask, junk_images)

        return mask
