import collections
import numpy as np
import h5py
from evaluation import Evaluation


class AttributeDataset(object):
    @property
    def info(self):
        return self._info

    def __init__(self, name, data, header, info):
        self.name = name
        self.data = data
        self.header = header
        assert isinstance(self.data[0], collections.Mapping)
        self._info = info
        self.attributes = info['attributes']

    def __getitem__(self, label):
        return self.data[label]

    def __len__(self):
        return len(self.data)

    def get_evaluation(self, model, num_copies):
        return AttributeEvaluation(model, self, num_copies)


class AttributeEvaluation(Evaluation):
    def __init__(self, model, attribute_dataset, num_copies):
        self.dataset = attribute_dataset
        name = self.dataset.name
        self.num_copies = num_copies
        super().__init__(name)

    def before_saving(self, endpoints, data):
        data_to_write = {}
        for attribute in self.dataset.attributes:
            output = endpoints['out_' + attribute].cpu().numpy()
            if self.num_copies > 1:
                # data is copied along batch dim
                nc = self.num_copies
                output = output.reshape(
                    (output.shape[0] // nc, nc) + output.shape[1:]
                )
            data_to_write[attribute] = output
            # reshape endpoint data?
            data_to_write['gt_' + attribute] = data[attribute].cpu().numpy()

        data_to_write['pid'] = data['pid'].cpu().numpy()
        return data_to_write

    def get_writer(self):
        raise NotImplementedError
    @staticmethod
    def _score(attribute_data):
        attributes = set()
        for attribute in attribute_data:
            if attribute.startswith('gt_'):
                continue
            attributes.add(attribute)

        results = {attribute: 0 for attribute in attributes}
        for attribute in attributes:
            gts = np.asarray(attribute_data['gt_' + attribute])
            preds = np.asarray(attribute_data[attribute])
            # no augmentation supported
            assert preds.ndim <= 3
            if preds.ndim == 3:
                preds = np.mean(preds, axis=1)
            preds = np.argmax(preds, axis=1)
            results[attribute] = np.sum(gts == preds)

        # Average is calculated according to the evaluation script
        average = 0
        for attribute in attributes:
            data = attribute_data[attribute]
            results[attribute] /= data.shape[0]
            average += results[attribute]

        average /= len(attributes)
        results['avg'] = average
        return results


    def score(self):
        with h5py.File(self.output_file, 'r') as embeddings:
            # uses rewritten pid
            return self._score(self.dataset.attributes, embeddings)


