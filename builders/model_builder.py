import torch.utils.model_zoo as model_zoo
import torch
from logger import get_logger
from models import load_model_class, BaseModel
from utils import format_dict_keys
logger = get_logger()


def get_model_dic_from_zoo(url):
    return model_zoo.load_url(url)


def get_model_dic_from_file(file, map_location=None):
    checkpoint = torch.load(file, map_location=map_location)
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def duplicate_layer(model_dict, from_layer, to_layers):
    from_dict = {k: v for k, v in model_dict.items() if k.startswith(from_layer)}
    if not from_layer.endswith('.'):
        from_layer = from_layer + '.'
    for to_layer in to_layers:
        for key, value in from_dict.items():
            if not to_layer.endswith('.'):
                to_layer = to_layer + '.'
            new_key = "{}{}".format(to_layer, key[len(from_layer):])
            logger.debug("%s => %s", key, new_key)
            model_dict[new_key] = value


def filter(dic, skips):
    for skip in skips:
        skipped_values = [k for k in dic.keys() if k.startswith(skip)]
        logger.debug(skipped_values)
    for skip in skips:
        dic = {k: v for k, v in dic.items() if not k.startswith(skip)}
    return dic


def set_weights(model, update_dict):
    if len(update_dict) == 0:
        return model
    model_dict = model.state_dict()

    # only update existing keys from the restored model that exist also in the current model
    # otherwise an error is thrown that this key does not exist in the current model
    model_dict.update((k, update_dict[k]) for k in model_dict.keys() & update_dict.keys())
    not_restored =  model_dict.keys() - update_dict.keys()
    print(not_restored)
    model.load_state_dict(model_dict)
    return model


def clean_dict(dic):
    """Removes module from keys. This is done when because of DataParallel!
    TODO directly init dataparallel?"""
    fresh_dict = {}
    prefix = "module."
    for key, value in dic.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        fresh_dict[key] = value
    return fresh_dict


def build(cfg):
    model_class = load_model_class(cfg['name'])
    model, skips, duplicate = model_class.build(cfg)
    # assert isinstance(model, BaseModel), "Make sure to inherit from BaseModel"
    pretrained = cfg.get('pretrained', False)
    init_from_file = cfg.get('init_from_file', None)
    init_from_dict = cfg.get('init_from_dict', None)
    update_dict = {}

    if 'weights' in cfg:
        update_dict = cfg['weights']
        logger.info('Restoring from passed weights')
    elif init_from_dict is not None:
        update_dict = init_from_dict
        logger.info('Restoring from checkpoint weights')
    elif init_from_file is not None and init_from_file != '':
        map_location = cfg.get('map_location')
        file_model_dict = get_model_dic_from_file(init_from_file, map_location)
        file_model_dict = clean_dict(file_model_dict)
        for from_layer, to_layers in duplicate:
            duplicate_layer(file_model_dict, from_layer, to_layers)
        #file_model_dict = filter(file_model_dict, skips)

        update_dict = file_model_dict
        logger.info("Restoring from file %s", init_from_file)
    elif pretrained:
        logger.info("Trying to restore from zoo")
        # TODO clean differentiate models and backbones
        if hasattr(model, 'model_urls'):
            if pretrained in model.model_urls:
                url = model.model_urls[pretrained]
            else:
                raise RuntimeError("No url named {} in {}.".format(pretrained, format_dict_keys(model.model_urls)))
        else:
            raise RuntimeError("No zoo urls available for model {}".format(cfg['name']))

        pretrained_dict = get_model_dic_from_zoo(url)
        for from_layer, to_layers in duplicate:
            duplicate_layer(pretrained_dict, from_layer, to_layers)
        pretrained_dict = filter(pretrained_dict, skips)
        update_dict = pretrained_dict
    else:
        logger.warning("Training from scratch")


    update_dict = clean_dict(update_dict)

    model = set_weights(model, update_dict)
    if 'seg_class_mapping' in cfg:
        mapping = cfg['seg_class_mapping']
    else:
        mapping = None
    model.seg_mapping = mapping

    if 'seg_class_mapping' in cfg:
        model.seg_mapping = model_cfg['seg_class_mapping']
    else:
        model.seg_mapping = None

    return model
