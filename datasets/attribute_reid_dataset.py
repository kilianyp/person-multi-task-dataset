import torch.utils.data
from evaluation import Evaluation
from writers.h5 import DynamicH5Writer
import os
import numpy as np
import h5py


class AttributeReidDataset(torch.utils.data.Dataset):
    @property
    def info(self):
        return {**self.attribute_dataset.info, **self.reid_dataset.info}

    def __init__(self, attribute, reid):
        super().__init__()
        self.attribute_dataset = attribute
        self.reid_dataset = reid
        self.data = reid.data
        self.header = {}
        self.header.update(reid.header)
        self.header.update(attribute.header)
        self.num_labels = reid.num_labels

    def __getitem__(self, index):
        data = self.reid_dataset[index]
        attribute_data = self.attribute_dataset[data['pid']]
        data.update(attribute_data)
        return data

    def __len__(self):
        return len(self.reid_dataset)

    def get_evaluation(self, model):
        raise RuntimeError("Use AttributeReidTestDataset for evaluation")


class AttributeReidTestDataset(torch.utils.data.Dataset):
    @property
    def info(self):
        return {**self.attribute_dataset.info, **self.reid_dataset.info}

    def __init__(self, attribute, reid):
        super().__init__()
        self.reid_dataset = reid
        self.attribute_dataset = attribute
        self.data = reid.data
        self.header = {}
        self.header.update(reid.header)
        self.header.update(attribute.header)
        self.label_dic, self.num_labels = get_test_label_dic(self.reid_dataset)
        print("num labels", self.num_labels)

    def __getitem__(self, index):
        data = self.reid_dataset[index]
        # attribute dataset expects rewritten pid
        # rewriting is possible
        if data['pid'] in [-1, 0]:
            # we need to write some data
            # this will be later ignored
            mapped = 0
        else:
            mapped = self.label_dic[data['pid']]
        attribute_data = self.attribute_dataset[mapped]
        data.update(attribute_data)
        return data

    def __len__(self):
        return len(self.reid_dataset)

    def get_evaluation(self, model):
        return AttributeReidEvaluation(model, self)


def get_test_label_dic(reid_test_dataset):
    # rewrite reid test dataset pids
    # this is necessary for attribute evaluation
    # because they don't provide the pid
    query_data = reid_test_dataset.query_data
    new_label = 0
    label_dic = {}
    for d in query_data:
        if d['pid'] in [-1, 0]:
            continue
        if d['pid'] not in label_dic:
            label_dic[d['pid']] = new_label
            new_label += 1

    return label_dic, new_label


from datasets.attribute_dataset import AttributeEvaluation
class AttributeReidEvaluation(Evaluation):
    def __init__(self, model, attribute_reid_dataset):
        self.reid_evaluation = attribute_reid_dataset.reid_dataset.get_evaluation(model)
        self.num_copies = self.reid_evaluation.num_copies
        self.attribute_evaluation = attribute_reid_dataset.attribute_dataset.get_evaluation(model, self.num_copies)
        self.dataset = attribute_reid_dataset
        name = self.dataset.reid_dataset.name
        super().__init__(name)

    @staticmethod
    def _score_attribute(attribute_data, pids, len_test):
        # only on gallery data
        for attribute, data in attribute_data.items():
            attribute_data[attribute] = data[:len_test]

        pids = pids[:len_test]

        # ignore junk_data 
        junk_mask = np.logical_and(pids != 0, pids != -1)
        for attribute, data in attribute_data.items():
            attribute_data[attribute] = data[junk_mask]

        return AttributeEvaluation._score(attribute_data)

    def score(self):
        scores = {}
        with h5py.File(self.output_file, 'r') as f:
            attribute_data = {}

            attributes = self.dataset.attribute_dataset.attributes
            for attribute in attributes:
                attribute_data[attribute] = np.asarray(f[attribute])
                attribute_data['gt_' + attribute] = np.asarray(f['gt_' + attribute])
            pids = np.asarray(f['pid'])
            pids = pids.astype(np.int32)
            attribute_score = self._score_attribute(attribute_data, pids, len(self.dataset.reid_dataset.test_data))
            scores.update(attribute_score)
        self.reid_evaluation.output_file = self.output_file
        reid_score = self.reid_evaluation.score()
        scores.update(reid_score)

        return scores

    def before_infere(self, data):
        return self.reid_evaluation.before_infere(data)

    def before_saving(self, endpoints, data):
        attribute_data = self.attribute_evaluation.before_saving(endpoints, data)
        reid_data = self.reid_evaluation.before_saving(endpoints, data)
        return {**attribute_data, **reid_data}

    def get_writer(self, output_path):
        self.output_path = output_path
        self.output_file = os.path.join(output_path, self.name + '.h5')
        self.writer = DynamicH5Writer(self.output_file)
        return self.writer

