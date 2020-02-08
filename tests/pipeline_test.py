import pytest
from losses.dummy import DummyLoss
from datasets.dummy import DummyDataset, create_dummy_data

from builders import (dataset_builder, scheduler_builder, dataloader_builder,
                      model_builder, optimizer_builder, loss_builder, evaluation_builder,
                      config_builder)
from torch.autograd import Variable
import torch
import json
from builders.config_builder import build_config
import logger as log

def test_train_cfg(cfg_file):
    with open(cfg_file) as f:
        cfg = json.load(f)

    cfg = build_config(cfg)
    print(cfg)
    train_cfg = cfg['training']
    dataloader_cfg = train_cfg['dataloader']
    model_cfg = train_cfg['model']
    optimizer_cfg = train_cfg['optimizer']
    loss_cfg = train_cfg['losses']
    scheduler_cfg = train_cfg['scheduler']
    device = torch.device('cuda')
    dataloader = dataloader_builder(dataloader_cfg)
    dataset = dataloader.dataset
    model = model_builder.build(model_cfg, dataset.info)


    optimizer = optimizer_builder.build(optimizer_cfg, model.parameters())


    #optimizer = torch.optim.SGD(model.parameters(), lr=eps0, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = scheduler_builder.build(scheduler_cfg, optimizer)
    loss = loss_builder.build(loss_cfg)
    file_logger = log.get_file_logger()
    model = torch.nn.DataParallel(model)
    # new experiment
    model = model.train()
    trained_models = []
    while lr_scheduler.run:
        lr_scheduler.step()
        for batch_id, (data, split_info) in enumerate(dataloader):
            #print(data)
            optimizer.zero_grad()
            data['imgs'] = data['img'].to(device)
            print("imgs", data['imgs'])
            imgs = Variable(data['img'], requires_grad=True)
            endpoints = model(imgs, model.module.endpoints)
            # threoretically losses could also be caluclated distributed.
            losses = loss(endpoints, data, split_info)
            print("losses", losses)
            print(torch.mean(losses))
            loss_mean = torch.mean(losses)
            loss_mean.backward()
            optimizer.step()
        break
    path = file_logger.save_checkpoint(model, optimizer, lr_scheduler.last_epoch)
    if path:
        trained_models.append(path)
    file_logger.close()


def test_evaluation_cfg():
    with open('./configs/market_evaluate.json') as f:
        cfg = json.load(f)
    cfg = build_config(cfg)
    evaluation_cfg = cfg['evaluation']
    dataloaders, model_cfgs = evaluation_builder.build(evaluation_cfg)
    with torch.no_grad():
        for model_cfg in model_cfgs:
            model = model_builder.build(model_cfg)
