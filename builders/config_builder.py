import sys
from sacred.config import ConfigScope
import json

def build_cfg_fn(name):
    module = sys.modules[__name__]
    name = name.lower()
    try:
        return getattr(module, '{}_cfg_fn'.format(name))
    except Exception as e:
        print(e)

        return no_default_cfg_fn


def no_default_cfg_fn():
    pass


def adam_cfg_fn():
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 0
    amsgrad = False


def sgd_cfg_fn():
    lr = 1e-3
    momentum = 0.0
    weight_decay = 0.0
    dampening = 0.0
    nesterov = False


def hermans_scheduler_cfg_fn():
    """In defense of the triplet_loss. Needs update every step!

    Assuming Market with 751 Pids, trained with P=18.
    This means one epoch corresponds to u_epoch = 751 / 18 = 41.722 updates.
    Therefore:

    t0 = 15000 / u_epoch = 360
    t1 = 25000 / u_epoch = 600

    """
    name = "exponential_decay"
    t1 = 600
    t0 = 360
    eps0 = 1e-4
    factor = 0.001


def huanghoujing_scheduler_cfg_fn():
    name = "exponential_decay"
    t1 = 300
    t0 = 151
    lr = 2e-4
    factor = 0.001


def mgn_scheduler_cfg_fn():
    """Schedule according to "Learning Discriminative Features with 
    Multiple Granularities for Person Re-Identification"""
    name = "multi_step"
    lr = 1e-2
    milestones = [40, 60]
    gamma = 0.1
    epochs = 80


def test_scheduler_cfg_fn():
    name = "test"
    epochs = 2


def pk_sampler_cfg_fn():
    P = 18
    K = 4


def trinet_cfg_fn():
    dim = 256


def batchhard_cfg_fn():
    margin = 'soft'


def general_cfg_fn():
    device_id = None

def train_cfg_fn():
    # After how many iterations a new checkpoint is created.")
    checkpoint_frequency = 100
    restore_checkpoint = None


def dataset_cfg_fn():
    loader_fn = "cv2"


def evaluation_dataset_cfg_fn():
    num_workers = 0


def evaluation_cfg_fn():
    delete = True


def reid_evaluation_cfg_fn():
    metric = "euclidean"
    gallery_batch_size = 10
    query_batch_size = 10


def pretty(dic):
    print(json.dumps(dic, indent=1))


def build_config(config):
    def build_default_config(name, config):
        fn = build_cfg_fn(name)
        scope = ConfigScope(fn)
        return scope(preset=config)

    def model_default_config(config):
        name = config['name']
        return build_default_config(name, config)

    def transform_default_config(config):
        return build_default_config('transform', config)

    def dataset_default_config(config):
        if isinstance(config, list):
            configs = []
            for c in config:
                configs.append(dataset_default_config(c))
            return configs
        scope = ConfigScope(dataset_cfg_fn)
        config = scope(preset=config)
        config['transform'] = transform_default_config(config.get('transform', {}))
        return config

    def single_sampler_default_config(config):
        name = config['type']
        config['dataset'] = dataset_default_config(config['dataset'])
        return build_default_config(name, config)

    def sampler_default_config(config):
        if "samplers" in config:
            #multi sampler
            samplers = config['samplers']
            for name, c in samplers.items():
                samplers[name] = single_sampler_default_config(c)
            return config
        else:
            return single_sampler_default_config(config)

    def loss_default_config(config):
        if isinstance(config, list):
            configs = []
            for c in config:
                configs.append(loss_default_config(c))
            return configs
        name = config['name']
        cfg = build_default_config(name, config)
        return cfg

    def scheduler_default_config(config):
        if 'preset' in config:
            name = "{}_scheduler".format(config['preset'])
        else:
            name = "{}_scheduler".format(config['name'])
        return build_default_config(name, config)


    def optimizer_default_config(config):
        name = config['name']
        return build_default_config(name, config)

    def training_default_config(config):
        config = build_default_config('train', config)
        for key, values in config.items():
            if key == 'model':
                config['model'] = model_default_config(config['model'])
            elif key == 'dataloader':

                config['dataloader']['sampler'] = sampler_default_config(config['dataloader']['sampler'])
            elif key == 'losses':
                config['losses'] = loss_default_config(config['losses'])
            elif key == 'scheduler':
                config['scheduler'] = scheduler_default_config(config['scheduler'])
            elif key == 'optimizer':
                config['optimizer'] = optimizer_default_config(config['optimizer'])
            elif key == 'checkpoint_frequency':
                pass
            elif key == 'restore_checkpoint':
                pass
            elif key == 'epochs':
                pass
            elif key == 'num_workers':
                pass
            else:
                raise ValueError(key)
        return config

    # TODO make central or get rid
    reid_datasets = ["market-1501", "duke"]
    reid_attribute_datasets = ["market-1501-attribute", "duke-attribute"]
    pose_datasets = ["mpii"]

    def evaluation_dataset_default_config(config):
        if isinstance(config, list):
            configs = []
            for c in config:
                configs.append(evaluation_dataset_default_config(c))
            return configs
        name = config['name'].lower()
        config = build_default_config('evaluation_dataset', config)
        if name in reid_datasets:
            return build_default_config('reid_evaluation', config)
        elif name in reid_attribute_datasets:
            return build_default_config('reid_attribute_evaluation', config)
        elif name in pose_datasets:
            return config
        else:
            raise ValueError("Unknown evaluation dataset in config builder: {}.".format(name))

    def evaluation_default_config(config):
        if 'experiment' in config:
            # TODO restore the config
            return config
        #config['sampler']['datasets'] = evaluation_dataset_default_config(config['sampler']['datasets'])
        config = build_default_config('evaluation', config)

        return config


    scope = ConfigScope(general_cfg_fn)
    config = scope(preset=config)
    for key, value in config.items():
        if key == 'training':
            config['training'] = training_default_config(config['training'])
        elif key == 'evaluation':
            config['evaluation'] = evaluation_default_config(config['evaluation'])
        elif key == 'validation':
            pass
        elif key == 'device_id':
            pass
        elif key == 'num_workers':
            pass
        elif key == 'restore_checkpoint':
            pass
        elif key == 'experiment':
            pass
        else:
            pass
    return config


def build(experiment):

    @experiment.config_hook
    def fill_with_default_config(config, command_name, logger):
        return build_config(config)
    return experiment
