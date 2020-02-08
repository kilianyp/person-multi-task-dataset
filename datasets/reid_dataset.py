from datasets.dataset import Dataset, ConcatDataset
from .utils import HeaderItem
import numpy as np
import h5py
import itertools
from evaluation import Evaluation
from writers.h5 import DynamicH5Writer
import os
from metrics import reid_score
from abc import abstractmethod
from logger import get_logger


def make_pid_dataset(csv_file, data_dir, limit=None):
    """Reads in a csv file according to the scheme "target, path".
    """
    from datasets.utils import make_dataset_unamed
    header = {"pid": HeaderItem((), -1), "path": HeaderItem((), '')}
    def default_name_fn(row):
        return row['path']

    data, header, dataset_info = make_dataset_unamed(csv_file, data_dir, default_name_fn, header)
    if not limit is None:
        new_data = []
        pids = set()
        for d in data:
            pid = d['pid']
            pids.add(pid)
            if len(pids) > limit:
                continue
            new_data.append(d)
        logger = get_logger()
        logger.info('Using %d/%d pids.', limit, len(pids))
        data = new_data

    return data, header, dataset_info

def rewrite_pids(data, start_label=0, prefix=None):
    """Rewrites pids in data to an ascedning sequence.
    Start_label: First pid becomes start_label.
    Prefix: Prefix added to the pid.
    """

    new_label = start_label
    label_dic = {}
    for x in data:
        pid = x['pid']
        if prefix:
            pid = "{}{}".format(prefix, pid)

        if not pid in label_dic:
            label_dic[pid] = new_label
            new_label += 1
        x['original_pid'] = pid
        x['pid'] = label_dic[pid]
    return new_label, label_dic


class ConcatReidDataset(ConcatDataset):
    @property
    def labels(self):
        return self.label_dic

    @property
    def data(self):
        # needed for rewrite pids
        # otherwise every image would be loaded -> slow
        return itertools.chain.from_iterable([d.data for d in self.datasets])

    @property
    def num_labels(self):
        return len(self.label_dic)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        start_label = 0
        self.label_dic = {}
        for d in self.datasets:
            start_label, label_dic = rewrite_pids(d.data, start_label, d.name)
            self.label_dic.update(label_dic)


class ReidDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels, self.label_dic = rewrite_pids(self.data)
        self.info['num_classes'] = self.num_labels
        self._header['original_pid'] = HeaderItem((), -1)

    def get_evaluation(self):
        raise RuntimeError("Use ReidTestDataset for evaluation")


class ReidTestDataset(Dataset):
    def __init__(self, name, metric, test_data, query_data, *args, **kwargs):
        super().__init__(name, test_data + query_data, *args, **kwargs)
        self.test_data = test_data
        self.query_data = query_data
        self.metric = metric

    def __getitem__(self, index):
        data = self.data[index]
        copied = data.copy()
        img = self.loader_fn(copied['path'])
        if self.transform is not None:
            self.transform.to_deterministic()
            img = self.transform.augment_image(img)
        copied['img'] = img
        return copied

    @abstractmethod
    def get_evaluation(self, model):
        pass


class ReidEvaluation(Evaluation):
    """Bound to h5"""
    def __init__(self, dataset, calc_dist, matcher_cls):
        super().__init__(dataset.name)
        self.dataset = dataset
        self.name = dataset.name
        self.num_copies = self.dataset.transform.num_copies
        
        def _score(gallery_emb, gallery_pids, gallery_fids,
                   query_emb, query_pids, query_fids):

            if self.num_copies > 1:
                query_emb = query_emb.mean(axis=1)
                gallery_emb = gallery_emb.mean(axis=1)

            dist_mat = calc_dist(query_emb, gallery_emb)

            matcher = matcher_cls(gallery_fids)

            mean_ap, cmc = reid_score(dist_mat,
                                      query_pids, query_fids,
                                      gallery_pids, gallery_fids,
                                      matcher)
            return {"map": mean_ap, "top1": cmc[0], "top2": cmc[1], "top5": cmc[4], "top10": cmc[9]}

        self._score = _score


    def get_writer(self, output_path):
        self.output_path = output_path
        self.output_file = os.path.join(output_path, self.name + '.h5')

        self.writer = DynamicH5Writer(self.output_file)
        return self.writer

    def before_infere(self, data):
        if self.num_copies > 1:
            self.b, self.n, c, h, w = data['img'].shape
            data['img'] = data['img'].view(-1, c, h, w)
        return data

    def before_saving(self, endpoints, data):
        data_to_write = {}
        emb = endpoints['emb']
        if self.num_copies > 1:
            # data is copied along batch dim
            emb = emb.reshape(
                (self.b, self.n) + emb.shape[1:]
            )
        data_to_write['emb'] = emb.cpu().numpy()

        data_to_write['pid'] = data['pid'].cpu().numpy()
        data_to_write['path'] = data['path']
        return data_to_write
    
    def score(self):
        # Load the two datasets
        with h5py.File(self.output_file, 'r') as f:
            emb = np.array(f['emb'])
            pids = np.array(f['pid'])
            pids = pids.astype('|U')
            fids = np.array(f['path'])
            fids = fids.astype('|U')

            gallery_emb = emb[:len(self.dataset.test_data)]
            gallery_pids = pids[:len(self.dataset.test_data)]
            gallery_fids = fids[:len(self.dataset.test_data)]

            query_emb = emb[len(self.dataset.test_data):]
            query_pids = pids[len(self.dataset.test_data):]
            query_fids = fids[len(self.dataset.test_data):]

            assert len(query_emb) == len(self.dataset.query_data)
            return self._score(gallery_emb, gallery_pids, gallery_fids,
                    query_emb, query_pids, query_fids)


