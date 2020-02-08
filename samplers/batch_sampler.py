import torch.utils.data.sampler as sampler

class BatchSampler(sampler.BatchSampler):
    def __init__(self, *args):
        super().__init__(*args)
        self.dataset = self.sampler.dataset

    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            batch.append(next(self.sampler))
        if len(batch) == self.batch_size:
            return batch
        if len(batch) > 0 and not self.drop_last:
            return batch
    def reset(self):
        self.sampler.reset()
