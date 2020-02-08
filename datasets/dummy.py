from datasets.utils import HeaderItem
from datasets import register_dataset
from datasets.dataset import Dataset


def create_dummy_pid_data(size, pids, name=''):
    imgs = []
    assert size > pids
    for i in range(size):
        pid = i % pids
        data = {'pid': pid,
                'dataset': name,
                'path': name}
        imgs.append(data)
    return imgs, {'pid': HeaderItem((), -1)}, {}


def create_dummy_data(size, name=''):
    imgs = []
    for i in range(size):
        data =  {'path': name,
                'dataset': name}
        imgs.append(data)
    return imgs, {}, {}


@register_dataset('dummy')
class DummyDataset(Dataset):
    def __init__(self, name, data, header, info):
        super().__init__(name, data, header, info)
        self.num_labels = None

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build(data_dir, cfg, *args, **kwargs):
        size = cfg['size']
        id = cfg.get('id', 'dummy')
        if 'num_pids' in cfg:
            num_pids = cfg['num_pids']
            data, header, info = create_dummy_pid_data(size, num_pids, id)
            dataset = DummyDataset(id, data, header, info)
        else:
            data, header, info = create_dummy_data(size, id)
            dataset = DummyDataset(id, data, header, info)
        return dataset


@register_dataset('dummy_reid')
class DummyReidDataset(DummyDataset):
    @staticmethod
    def build(data_dir, cfg, *args, **kwargs):
        size = cfg['size']
        id = cfg.get('id', 'dummy')
        num_pids = cfg['num_pids']
        data, header, info = create_dummy_pid_data(size, num_pids, id)
        dataset = DummyReidDataset(id, data, header, info)
        return dataset
