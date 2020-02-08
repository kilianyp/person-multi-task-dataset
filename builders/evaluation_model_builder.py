import os
import torch
import json


def build(cfg):
    """
    Builds configurations that can be used by model builder for evaluation.
    """
    experiment = cfg.get('experiment')
    if experiment:
        return restore_from_experiment(experiment)

    files = cfg['files']
    model_cfgs = []
    for file in files:
        original_cfg = restore_from_file(file)
        # update values
        original_cfg.update(cfg)
        model_cfgs.append(original_cfg)
    return model_cfgs


def restore_from_files(files):
    model_cfgs = []
    for file in files:
        model_cfgs.append(restore_from_file(file))
    return model_cfgs


def restore_from_file(model_file):
    checkpoint = torch.load(model_file)
    # TODO pretty chaotic
    if 'model_cfg' in checkpoint:
        model_cfg = checkpoint['model_cfg']
        model_cfg['init_from_file'] = model_file
        return model_cfg

    # For backwards compability
    basepath = os.path.dirname(model_file)
    config_file = os.path.join(basepath, 'config.json')
    if not os.path.isfile(config_file):
        raise RuntimeError('Cannot restore from {}. Config.json not found'
                           .format(basepath))

    with open(config_file, 'r') as f:
        config = json.load(f)
    if 'seed' in config:
        # new config format
        model_cfg = config['training']['model']
    else:
        # old format
        model_cfg = {}
        model_cfg['name'] = config['model']
        model_cfg['dim'] = config['dim']
        model_cfg['num_branches'] = config['num_branches']
        model_cfg['num_classes'] = config['num_classes']

    model_cfg['pretrain'] = False
    model_cfg['init_from_file'] = model_file
    return model_cfg


def restore_from_experiment(experiment):
    raise NotImplemented
