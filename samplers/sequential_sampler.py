import torch.utils.data.sampler as sampler
from samplers.batch_sampler import BatchSampler
from samplers import register_single_sampler


@register_single_sampler('sequential')
class SequentialSampler(sampler.SequentialSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.n = 0

    def reset(self):
        self.n = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.n < len(self.dataset):
            n = self.n
            self.n += 1
            return n
        raise StopIteration

    @staticmethod
    def build(dataset, cfg):
        drop_last = cfg.get('drop_last', True)
        batch_size = cfg['batch_size']
        seq_sampler = SequentialSampler(dataset)
        return BatchSampler(seq_sampler, batch_size, drop_last)
