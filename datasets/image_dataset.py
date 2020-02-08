from datasets.dataset import Dataset

class FullImageDataset(Dataset):
    def __init__(self, name, data, header, info, *args, **kwargs):
        super().__init__(name, data, header, info, *args, **kwargs)

    def __getitem__(self, index):
        data = self.data[index]
        copied = data.copy()
        img = self.loader_fn(copied['path'])
        shape = img.shape
        copied['height'] = shape[0]
        copied['width'] = shape[1]
        if self.transform is not None:
            self.transform.to_deterministic()
            img = self.transform.augment_image(img)

        copied['img'] = img
        return copied
    @staticmethod
    def build(cfg):
        data, header, dataset_info = make_dataset_fn(source_file, data_dir, name, **dataset_args)
