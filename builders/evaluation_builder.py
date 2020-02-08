from builders import evaluation_model_builder
from builders import dataloader_builder
from logger import get_logger
logger = get_logger()


def build(cfg):
    #TODO: Make this consistent in terms of return vals
    evaluation_cfgs = dict()
    dataloaders = []
    for name, dataloader_cfg in cfg:
        dataloader = dataloader_builder.build(dataloader_cfg)
        dataloaders.append(dataloader)

    model_cfg = cfg['model']

    model_cfgs = evaluation_model_builder.build(cfg['model'])
    # overwrite restored values
    for model_cfg in model_cfgs:
        model_cfg.update(cfg['model'])
    return evaluation_cfgs, model_cfgs
