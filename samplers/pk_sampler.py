import numpy as np
from samplers import register_single_sampler


def create_pids2idxs(dataset):
    """Creates a mapping between pids and indexes of images for that pid.
    Returns:
        2D List with pids => idx
    """
    pid2idxs = {}
    for idx, data in enumerate(dataset.data):
        pid = data['pid']
        if pid not in pid2idxs:
            pid2idxs[pid] = [idx]
        else:
            pid2idxs[pid].append(idx)
    return pid2idxs


@register_single_sampler('pk')
class PKSampler(object):
    """Sampler to create batches with P x K.

       Only returns indices.

    """
    def __init__(self, P, K, dataset, drop_last=True):
        self.P = P
        self.K = K
        self.batch_size = self.P * self.K
        self.dataset = dataset
        self.drop_last = drop_last
        self.pid2idxs = create_pids2idxs(dataset)
        self.pids = list(self.pid2idxs.keys())
        self.completed_iter = 0
        self.reset()

    def __iter__(self):
        self.reset()
        return self

    def reset(self):
        self.p_idx = 0
        # TODO this is unstable when parts of the pis are mixed
        # datatypes like string an int
        self.P_perm = np.random.permutation(self.pids)

    def __next__(self):
        batch = []
        for _ in range(self.P):
            try:
                pid = self.P_perm[self.p_idx]
            except IndexError:
                if len(batch) > 1 and not self.drop_last:
                    return batch
                else:
                    self.completed_iter += 1
                    raise StopIteration
            images = self.pid2idxs[pid]
            K_perm = np.random.permutation(len(images))
            # fill up by repeating the permutation
            if len(images) < self.K:
                K_perm = np.tile(K_perm, self.K//len(images))
                left = self.K - len(K_perm)
                K_perm = np.concatenate((K_perm, K_perm[:left]))
            for k in range(self.K):
                batch.append(images[K_perm[k]])
            self.p_idx += 1

        return batch

    def __len__(self):
        if self.drop_last:
            return len(self.pid2idxs) // self.P
        else:
            return (len(self.pid2idxs) + self.P - 1) // self.P

    @staticmethod
    def build(dataset, cfg):
        drop_last = cfg.get('drop_last', True)
        P = cfg['P']
        K = cfg['K']
        return PKSampler(P, K, dataset, drop_last)


