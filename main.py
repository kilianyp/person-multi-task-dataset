import sys
import os
import time
import json
from settings import Config

path = os.path.abspath('./sacred')
sys.path = [path] + sys.path
import torch
from builders import (scheduler_builder, dataloader_builder,
                      model_builder, optimizer_builder, loss_builder,
                      config_builder)
import logger as log
from logger import report_after_batch, report_after_epoch, report_after_training
from sacred.observers import SlackObserver, FileStorageObserver, MongoObserver
from sacred import Experiment
from argparse import ArgumentParser
from utils import ExitHandler
import utils


def set_device(config):
    device_id = config['device_id']

    if not device_id == 'cpu' and torch.cuda.is_available():
        if device_id is None:
            device = torch.device('cuda')
        else:
            if isinstance(device_id, list):
                device = torch.device('cuda:{}'.format(device_id[0]))
            else:
                device = torch.device('cuda:{}'.format(device_id))

        torch.cuda.init()
    else:
        device = torch.device('cpu')
        # make compatible with torch dataparallel
        # TODO does not work with data parallel
        config['device_id'] = []
    config['device'] = device


def main(_run):
    # initialize logger after observers are appended
    log.initialize(_run)
    cfg = _run.config
    set_device(cfg)
    train_cfg = cfg['training']
    validation_cfg = cfg.get('validation')
    checkpoint_frequency = train_cfg['checkpoint_frequency']
    restore_checkpoint_cfg = train_cfg['restore_checkpoint']
    max_epochs = train_cfg['epochs']
    model_files = run_train(train_cfg['dataloader'], train_cfg['model'], train_cfg['scheduler'],
                            train_cfg['optimizer'], train_cfg['losses'], validation_cfg,
                            checkpoint_frequency, restore_checkpoint_cfg, max_epochs,
                            _run)
    if 'test' in cfg:
        test_dataset_cfg = cfg['test']
        score = evaluate_checkpoint_on(test_dataset_cfg, model_files[-1])
        log_result(score, _run)
        return format_result(score)
    else:
        return True


def run_train(dataloader_cfg, model_cfg, scheduler_cfg,
        optimizer_cfg, loss_cfg, validation_cfg, checkpoint_frequency,
        restore_checkpoint, max_epochs, _run):

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True
    exit_handler = ExitHandler()

    device = _run.config['device']
    device_id = _run.config['device_id']

    # during training just one dataloader
    dataloader = dataloader_builder.build(dataloader_cfg)[0]

    epoch = 0
    if restore_checkpoint is not None:
        model_cfg, optimizer_cfg, epoch = utils.restore_checkpoint(restore_checkpoint, model_cfg, optimizer_cfg)

    def overwrite(to_overwrite, dic):
        to_overwrite.update(dic)
        return to_overwrite

    # some models depend on dataset, for example num_joints
    model_cfg = overwrite(dataloader.dataset.info, model_cfg)
    model = model_builder.build(model_cfg)

    loss_cfg['model'] = model
    loss = loss_builder.build(loss_cfg)
    loss = loss.to(device)

    parameters = list(model.parameters()) + list(loss.parameters())
    optimizer = optimizer_builder.build(optimizer_cfg, parameters)

    lr_scheduler = scheduler_builder.build(scheduler_cfg, optimizer, epoch)

    if validation_cfg is None:
        validation_dataloaders = None
    else:
        validation_dataloaders = dataloader_builder.build(validation_cfg)
        keep = False

    file_logger = log.get_file_logger()
    logger = log.get_logger()


    model = torch.nn.DataParallel(model, device_ids=device_id)
    model.cuda()

    model = model.train()
    trained_models = []

    exit_handler.register(file_logger.save_checkpoint,
                          model, optimizer, "atexit",
                          model_cfg)

    start_training_time = time.time()
    end = time.time()
    while epoch < max_epochs:
        epoch += 1
        lr_scheduler.step()
        logger.info("Starting Epoch %d/%d", epoch, max_epochs)
        len_batch = len(dataloader)
        acc_time = 0
        for batch_id, data in enumerate(dataloader):
            optimizer.zero_grad()
            endpoints = model(data, model.module.endpoints)
            logger.debug("datasets %s", list(data['split_info'].keys()))

            data.update(endpoints)
            # threoretically losses could also be caluclated distributed.
            losses = loss(endpoints, data)
            loss_mean = torch.mean(losses)
            loss_mean.backward()
            optimizer.step()

            acc_time += time.time() - end
            end = time.time()

            report_after_batch(_run=_run, logger=logger, batch_id=batch_id, batch_len=len_batch,
                               acc_time=acc_time, loss_mean=loss_mean, max_mem=torch.cuda.max_memory_allocated())

        if epoch % checkpoint_frequency == 0:
            path = file_logger.save_checkpoint(model, optimizer, epoch, model_cfg)
            trained_models.append(path)

        report_after_epoch(_run=_run, epoch=epoch, max_epoch=max_epochs)

        if validation_dataloaders is not None and \
                epoch % checkpoint_frequency == 0:
            model.eval()

            # Lets cuDNN benchmark conv implementations and choose the fastest.
            # Only good if sizes stay the same within the main loop!
            # not the case for segmentation
            torch.backends.cudnn.benchmark = False
            score = evaluate(validation_dataloaders, model, epoch, keep=keep)
            logger.info(score)
            log_score(score, _run, prefix="val_", step=epoch)
            torch.backends.cudnn.benchmark = True
            model.train()

    report_after_training(_run=_run, max_epoch=max_epochs, total_time=time.time() - start_training_time)
    path = file_logger.save_checkpoint(model, optimizer, epoch, model_cfg)
    if path:
        trained_models.append(path)
    file_logger.close()
    # TODO get best performing val model
    evaluate_last = _run.config['training'].get('evaluate_last', 1)
    if len(trained_models) < evaluate_last:
        logger.info("Only saved %d models (evaluate_last=%d)", len(trained_models), evaluate_last)
    return trained_models[-evaluate_last:]


def evaluate_checkpoint(_run):
    log.initialize(_run)
    cfg = _run.config
    set_device(cfg)
    validation_cfg = cfg['validation']
    restore_checkpoint_cfg = cfg['restore_checkpoint']
    model_update_cfg = cfg.get('model', {})
    scores = evaluate_checkpoint_on(restore_checkpoint_cfg, validation_cfg, _run, model_update_cfg)
    log_score(scores, _run, "val_")
    return format_result(scores)


def evaluate_checkpoint_on(restore_checkpoint, dataset_cfg, _run, model_update_cfg={}):
    model_cfg, _, epoch = utils.restore_checkpoint(restore_checkpoint, model_cfg=model_update_cfg, map_location='cpu')
    #model_cfg['backbone']['output_dim'] = 256
    dataloaders = dataloader_builder.build(dataset_cfg)
    model = model_builder.build(model_cfg)
    # TODO needs to be from dataset
    if 'seg_class_mapping' in model_cfg:
        mapping = model_cfg['seg_class_mapping']
    else:
        mapping = None

    model.seg_mapping = mapping

    model = torch.nn.DataParallel(model, device_ids=_run.config['device_id'])
    model = model.cuda()
    return evaluate(dataloaders, model, epoch, keep=True)


def evaluate(dataloaders, model, epoch, keep=False):
    import utils
    import shutil
    scores = {}
    model = model.eval()
    file_logger = log.get_file_logger()
    path_prefix = os.path.join(file_logger.get_log_dir(), 'results')
    for dataloader in dataloaders:
        dataset = dataloader.dataset
        # this is a change in design, 
        # dataset creates evaluation
        evaluation = dataset.get_evaluation(model)
        print(evaluation.name)
        folder_name = os.path.join(path_prefix, str(epoch))
        utils.create_dir_recursive(folder_name)
        # evaluation creates writer
        with evaluation.get_writer(folder_name) as writer:
            for idx, data in enumerate(dataloader):
                data = evaluation.before_infere(data)
                endpoints = model.module.infere(data)
                data_to_write = evaluation.before_saving(endpoints, data)
                writer.write(**data_to_write)
                print("\rDone (%d/%d)" % (idx, len(dataloader)), flush=True, end='')

        score = evaluation.score()
        if evaluation.name in scores:
            scores[evaluation.name].update(score)
        else:
            scores[evaluation.name] = score
        print(score)


    if not keep:
        logger = log.get_logger()
        logger.warning("deleting evaluation files in %s", path_prefix)
        shutil.rmtree(path_prefix)

    return scores

def test(dataloaders, model, epoch):
    import utils
    model = model.eval()
    file_logger = log.get_file_logger()
    path_prefix = os.path.join(file_logger.get_log_dir(), 'results')
    for dataloader in dataloaders:
        dataset = dataloader.dataset
        # this is a change in design, 
        # dataset creates evaluation
        test_set = dataset.get_test(model)
        folder_name = os.path.join(path_prefix, str(epoch))
        utils.create_dir_recursive(folder_name)
        # evaluation creates writer
        for idx, data in enumerate(dataloader):
            test_set.write(data)
            print("\rDone (%d/%d)" % (idx, len(dataloader)), flush=True, end='')



def test_config(_run):
    pass
    # logger = log.get_logger()
    # cfg = _run.config
    # set_device(cfg)
    # train_cfg = cfg['training']
    # checkpoint_frequency = train_cfg['checkpoint_frequency']
    # restore_checkpoint_cfg = train_cfg['restore_checkpoint']
    #
    # dataloader_cfg = train_cfg['dataloader']
    # model_cfg = train_cfg['model']
    # scheduler_cfg = train_cfg['scheduler']
    # optimizer_cfg = train_cfg['optimizer']
    # loss_cfg = train_cfg['losses']
    #
    # device = _run.config['device']
    # device_id = _run.config['device_id']
    #
    # dataloader = dataloader_builder.build(dataloader_cfg)
    #
    # model_cfg_appendix, optimizer_cfg_appendix, epoch, _ = restore_checkpoint(restore_checkpoint_cfg)
    # model_cfg.update(model_cfg_appendix)
    # optimizer_cfg.update(optimizer_cfg_appendix)
    #
    # def overwrite(to_overwrite, dic):
    #     to_overwrite.update(dic)
    #     return to_overwrite
    #
    # model_cfg = overwrite(dataloader.dataset.info, model_cfg)
    # model = model_builder.build(model_cfg)
    #
    # loss = loss_builder.build(loss_cfg)
    #
    # parameters = list(model.parameters()) + list(loss.parameters())
    # optimizer = optimizer_builder.build(optimizer_cfg, parameters)
    #
    #
    # validation_cfg = cfg.get('validation')
    # if validation_cfg is None:
    #     validation_dataloaders = None
    #     logger.warning("No validation given")
    # else:
    #     validation_dataloaders = dataloader_builder.build(validation_cfg)
    #
    # if 'evaluation' in cfg:
    #     evaluation_cfg = cfg['evaluation']
    #     dataloaders, model_cfgs = evaluation_builder.build(evaluation_cfg)
    #     delete = evaluation_cfg['delete']
    # else:
    #     logger.warning("No evaluation given")
    #
    # logger.info("Success")


def evaluate_experiment(_run):
    log.initialize(_run)
    new_cfg = _run.config
    set_device(new_cfg)
    experiment = new_cfg['experiment']
    # quick fix for last saved model
    num_models = new_cfg.get('last_x', 1)
    model_paths = log.Logger.get_all_model_paths(experiment)
    model_paths = model_paths[-num_models:]

    cfg_path = log.Logger.get_cfg_path(experiment)
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    # overwrite cfg
    # WARNING this means we are using a config that has not been
    # filled with default values
    cfg.update(new_cfg)
    set_device(cfg)
    # TODO
    for model_path in model_paths:
        score = evaluate_checkpoint_on(model_path, new_cfg['validation'], _run)
        log_score(score, _run)
        print(format_result(score))
    return True


def show_options():
    from models import get_all_models
    from samplers import get_all_multi_samplers, get_all_single_samplers
    print("models")
    print((',\n').join(get_all_models()))
    print("single samplers")
    print((',\n').join(get_all_single_samplers()))
    print("multi samplers")
    print((',\n').join(get_all_multi_samplers()))


def format_result(result):
    formatted = "Results:\n"
    metrics = {}
    for eval_name, score in result.items():
        for metric, value in score.items():
            metric_name = "{} @{}".format(metric, eval_name)
            if metric_name in metrics:
                metrics[metric_name].append(value)
            else:
                metrics[metric_name] = [value]

    for metric_name, values in metrics.items():
        try:
            formatted += "{}: {}\n".format(metric_name, ','.join(list(values)))
        except:
            formatted += "{}: {}\n".format(metric_name, str(values))
    return formatted


def log_result(results, _run):
    for model, scores in results.items():
        for eval_name, score in scores.items():
            for metric, value in score.items():
                metric_name = "{} @{}".format(metric, eval_name)
                _run.log_scalar(metric_name, value)


def log_score(scores, _run, prefix="", step=None):
    for eval_name, score in scores.items():
        for metric, value in score.items():
            metric_name = prefix + "{} @{}".format(metric, eval_name)
            _run.log_scalar(metric_name, value, step)


def create_base_experiment(sacred_args, name=None): #path=Config.LOG_DIR, db_name=Config.DB_NAME):
    ex = Experiment(name)
    print(name)
    ex.capture(set_device)
    ex.main(main)
    ex.capture(run_train)
    ex.command(evaluate_experiment)
    ex.command(test_config)
    ex.command(show_options)
    ex.command(evaluate_checkpoint)

    # set default values
    ex = config_builder.build(ex)

    # set observers but check if maybe sacred will create them
    # on its own
    """ TODO
    Problem is that we create the experiment before the command line is parsed by sacred.
    But then we cannot set default values without using a shell script. Or
    modify the file path for the logger."""

    if Config.LOG_DIR is not None and '-F' not in sacred_args:
        path = Config.LOG_DIR
        if name is not None:
            path = os.path.join(path, name)
        file_ob = FileStorageObserver.create(path)
        ex.observers.append(file_ob)

    if Config.SLACK_WEBHOOK_URL != "":
        slack_ob = SlackObserver(Config.SLACK_WEBHOOK_URL)
        ex.observers.append(slack_ob)


    if (Config.MONGO_DB_NAME is not None and Config.MONGO_DB_NAME != "") \
            and '-m' not in sacred_args:
        if Config.MONGO_USER != "":
            mongo_ob = MongoObserver.create(username=Config.MONGO_USER, password=Config.MONGO_PW,
                                            url=Config.MONGO_URL, authMechanism="SCRAM-SHA-256", db_name=Config.MONGO_DB_NAME)
        else:
            mongo_ob = MongoObserver.create(url=Config.MONGO_URL, db_name=Config.MONGO_DB_NAME)
        ex.observers.append(mongo_ob)

    return ex


if __name__ == "__main__":
    parser = ArgumentParser()
    # the name is set in run information
    parser.add_argument('-n', default=None)
    args, unknown_args = parser.parse_known_args()

    ex = create_base_experiment(unknown_args, name=args.n)
    ex.run_commandline()
