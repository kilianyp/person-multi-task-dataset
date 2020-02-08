from losses import choices
from losses.multi_loss import (SingleLoss, MultiLoss, WeightModule,
                               DynamicFocalLoss, LinearWeightedLoss,
                               DynamicFocalLossModule, UncertaintyLoss,
                               UncertaintyLossNotNegative,
                               DynamicFocalKeyLoss, ParetoLoss)
from dataflow import DataFlowConfig, DataFlowController
from losses.triplet_loss import BatchHard, BatchHardAttention
from losses.softmax import Softmax, CrossEntropyLoss, BootstrappedCrossEntropyLoss
from losses.attribute import Attribute
from losses.regression import MSELoss, L1Loss, L2Loss


def build(cfg):
    type = cfg['type'].lower()
    if is_multi_loss(type):
        return build_multi_loss(type, cfg)
    else:
        return build_single_loss(type, cfg)


def build_loss(cfg):
    endpoint = cfg['endpoint']
    type = cfg['type'].lower()
    if type == 'batchhard':
        margin = cfg['margin']
        return _build_batch_hard_loss(endpoint, margin)
    elif type == 'batchhard_with_attention':
        margin = cfg['margin']
        return _build_batch_hard_loss_with_attention(endpoint, margin)
    elif type == 'softmax':
        target = cfg['target']
        return _build_softmax_loss(endpoint, target)
    elif type == 'attribute':
        attributes = endpoint
        return _build_attribute_loss(attributes)
    elif type == 'mse':
        target = cfg['target']
        return _build_mse_loss(endpoint, target)
    elif type == 'l1':
        target = cfg['target']
        return _build_l1_loss(endpoint, target)
    elif type == 'l2':
        target = cfg['target']
        return _build_l2_loss(endpoint, target)
    elif type == 'crossentropy':
        target = cfg['target']
        return _build_crossentropy_loss(endpoint, target)
    elif type == 'bootstrappedcrossentropy':
        return _build_bootstrapped_crossentropy_loss(cfg)

    raise ValueError("Loss {} does not exist. Please choose from {}.".format(type, ", ".join(choices)))


def build_controllers(cfgs):
    endpoint_cfgs = []
    data_cfgs = []
    for cfg in cfgs:
        output_name = cfg['name']
        e_cfg = DataFlowConfig(cfg['endpoint'], output_name)
        d_cfg = DataFlowConfig(cfg['dataset'], output_name)
        endpoint_cfgs.append(e_cfg)
        data_cfgs.append(d_cfg)

    return DataFlowController(endpoint_cfgs), DataFlowController(data_cfgs)


def is_multi_loss(name):
    multi_losses = ["DynamicFocalLoss".lower(),
                    "DynamicFocalKeyLoss".lower(),
                    "LinearWeightedLoss".lower(),
                    "UncertaintyLoss".lower(),
                    "UncertaintyLossNotNegative".lower(),
                    "pareto_loss".lower()]
    if name.lower() in multi_losses:
        return True
    return False


def build_multi_loss(type, cfg):
    # WARNING / TODO all information is stored in
    # a single dictionary. Danger of double keys

    losses = cfg['losses']
    if not isinstance(losses, list):
        if len(losses) < 2:
            print("Using multi_loss even though less than  two losses for MultiLoss.")

    if type == "DynamicFocalLoss".lower():
        delta = cfg['delta']
        weight_module = build_dynamic_focal_loss(cfg['losses'], delta)
    elif type == "DynamicFocalKeyLoss".lower():
        weight_module = build_dynamic_focal_key_loss(cfg['losses'])
    elif type == "UncertaintyLoss".lower():
        weight_module = build_uncertainty_loss(cfg['losses'])
    elif type == "UncertaintyLossNotNegative".lower():
        weight_module = build_uncertainty_loss_not_negative(cfg['losses'])
    elif type == "LinearWeightedLoss".lower():
        weight_module = build_linear_weighted_loss(cfg['losses'])
    elif type == "pareto_loss":
        model = cfg['model']
        weight_module = build_pareto_loss(cfg['losses'], model)
    else:
        raise ValueError("Loss {} does not exist. Please choose from {}.".format(type, ", ".join(choices)))

    endpoint_controller, data_controller = build_controllers(cfg['losses'])
    return MultiLoss(weight_module, data_controller)


def build_uncertainty_loss(loss_cfgs):
    losses = {}
    for cfg in loss_cfgs:
        loss_module = build_loss(cfg)
        name = cfg['name']
        log_sig_sq = cfg['log_sig_sq']
        losses[cfg['name']] = UncertaintyLoss(loss_module, name, log_sig_sq)

    return WeightModule(losses)


def build_uncertainty_loss_not_negative(loss_cfgs):
    losses = {}
    for cfg in loss_cfgs:
        loss_module = build_loss(cfg)
        name = cfg['name']
        losses[cfg['name']] = UncertaintyLossNotNegative(loss_module, name)

    return WeightModule(losses)


def build_dynamic_focal_loss(task_cfgs, delta):
    """According to "A Coarse-to-fine Pyramidal Model for Person Re-identification
    via Multi-Loss Dynamic Training" by Zheng et al.
    https://arxiv.org/pdf/1810.12193.pdf
    """
    if len(task_cfgs) != 2:
        raise RuntimeError("Expect triplet and id loss.")

    tr_task = None
    id_task = None
    for cfg in task_cfgs:
        task = (cfg['name'], build_dynamic_focal_task(cfg))
        if cfg['type'].lower() == "batchhard":
            tr_task = task
        elif cfg['type'].lower() == "softmax":
            id_task = task
        else:
            raise RuntimeError("Expect triplet and id loss.")

    if tr_task is None or id_task is None:
        raise RuntimeError("Expect triplet and id task.")
    return DynamicFocalLossModule(delta, tr_task, id_task)


def build_dynamic_focal_task(cfg):
    alpha = cfg['alpha']
    gamma = cfg['gamma']
    p0 = cfg['p0']
    loss_module = build_loss(cfg)
    name = cfg['name']
    return DynamicFocalLoss(alpha, gamma, p0, loss_module, name)


def build_dynamic_focal_key_task(cfg):
    alpha = cfg['alpha']
    gamma = cfg['gamma']
    loss_module = build_loss(cfg)
    name = cfg['name']
    return DynamicFocalKeyLoss(alpha, gamma, loss_module, name)


def build_dynamic_focal_key_loss(task_cfgs):
    """According to "Dynamic Task Prioritization for Multitask Learning"
    by Michelle Guo et al."""
    losses = {}
    for task_cfg in task_cfgs:
        name = task_cfg['name']
        losses[name] = build_dynamic_focal_key_task(task_cfg)

    return WeightModule(losses)


def build_linear_weighted_loss(loss_cfgs):
    losses = {}
    for cfg in loss_cfgs:
        weight = cfg['weight']
        loss_module = build_loss(cfg)
        losses[cfg['name']] = LinearWeightedLoss(weight, loss_module)

    return WeightModule(losses)


def build_pareto_loss(loss_cfgs, model):
    losses = {}
    for cfg in loss_cfgs:
        loss_module = build_loss(cfg)
        losses[cfg['name']] = loss_module
    return ParetoLoss(losses, model)


def build_single_loss(name, cfg):
    loss = build_loss(cfg)
    return SingleLoss(loss)


def _build_softmax_loss(endpoint, target):
    return Softmax(target_name=target, endpoint_name=endpoint)


def _build_batch_hard_loss(endpoint, margin):
    return BatchHard(margin, endpoint_name=endpoint)

def _build_batch_hard_loss_with_attention(endpoint, margin):
    return BatchHardAttention(margin, endpoint_name=endpoint)

def _build_attribute_loss(attributes):
    return Attribute(attributes)


def _build_mse_loss(endpoint, target):
    return MSELoss(endpoint_name=endpoint, target_name=target)


def _build_l1_loss(endpoint, target):
    return L1Loss(endpoint_name=endpoint, target_name=target)


def _build_l2_loss(endpoint, target):
    return L2Loss(endpoint_name=endpoint, target_name=target)


def _build_crossentropy_loss(endpoint, target):
    return CrossEntropyLoss(endpoint_name=endpoint, target_name=target)

def _build_bootstrapped_crossentropy_loss(cfg):
    endpoint = cfg['endpoint']
    target = cfg['target']
    top_k_percent = cfg['top_k_percent']
    hard_mining_step = cfg['hard_mining_step']
    return BootstrappedCrossEntropyLoss(target, endpoint,
            top_k_percent_pixels=top_k_percent, hard_mining_step=hard_mining_step)
