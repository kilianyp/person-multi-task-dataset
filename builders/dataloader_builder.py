import torch
from datasets.dataset import MultiDataset
from datasets.utils import HeaderItem
import functools
from builders import sampler_builder
from builders import dataset_builder
from torch.utils.data.dataloader import default_collate
from dataloader import _collate_fn

def build(cfg):
    if isinstance(cfg, list):
        dataloaders = []
        for c in cfg:
            dataloaders.extend(build(c))
        return dataloaders
    else:
        num_workers = cfg.get('num_workers', 4)
        pin_memory = cfg.get('pin_memory', True)
        print("num_workers", num_workers)
        if num_workers == -1:
            import multiprocessing
            # one for main thread
            num_workers = multiprocessing.cpu_count() - 1
            print("num_workers", num_workers)

        if 'sampler' in cfg:
            sampler, dataset = sampler_builder.build(cfg['sampler'])
            if isinstance(dataset, MultiDataset):
                collate_fn = build_collate_fn(dataset.header)
            else:
                collate_fn = default_collate

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                num_workers=num_workers, pin_memory=pin_memory
            )

        else:
            # use default
            dataset = dataset_builder.build(cfg['dataset'])
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.get('batch_size', 10),
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        return [dataloader]


ERROR_STRING = 'ERROR_should_always_be_set'


def build_collate_fn(keys, sanity_check=False):
    if 'path' not in keys:
        keys['path'] = HeaderItem((), ERROR_STRING)
    if 'img' not in keys:
        keys['img'] = HeaderItem((), ERROR_STRING)

    return functools.partial(_collate_fn, keys=keys, sanity_check=sanity_check)
