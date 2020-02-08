from datasets.dataset import Dataset
import torch.tensor
import numpy as np


class SegmentationDataset(Dataset):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, index):

        datum = self.data[index]
        copied = datum.copy()

        img = np.array(self.loader_fn(copied['path']).convert("RGB"))
        seg = np.array(self.loader_fn(copied['gt-path']))

        if self.transform is not None:
             self.transform.to_deterministic()
             img, orig_img = self.transform.augment_image(img, return_unnormalized=True)
             seg = self.transform.augment_segmentation(seg, self.info['raw_classes'])

        if 'conversion' in self.info:
             seg = self.info['conversion'][seg]

        copied['gt-seg'] = torch.from_numpy(seg)
        copied['img'] = img
        copied['orig-img'] = orig_img

        return copied
