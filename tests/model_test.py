import pytest
from builders import model_builder
from models import get_all_models
import sys
import torch


attribute_cfg = {
    "attributes": {
        "gender": 2,
        "age": 2,
        "backpack": 2
    },
    "dropout": True,
    "name": "attribute"
}


resnet_cfg = {
    "name": "resnet",
    "stride": 1
}

resnet_groupnorm_cfg = {
    "name": "resnet_groupnorm",
    "stride": 1,
    "ncpg": 16,
    "pretrained": False
}
baseline_cfg = {
    "backbone": resnet_cfg,
    "pooling": "max"
}


classification_cfg = {
    "num_classes": 1,
    "merging_block": {
        "name": "single",
        "endpoint": "softmax"
    }
}


classification_triplet_cfg = {
    "num_classes": 1
}


conv4_multi_task_cfg = {
    "tasks": {
        "reid": {"pooling": "max"},
        "pose": {"num_joints": 16}

    }
}

conv4_2_head_batch_cfg = {
    "stride": 1,
    "pretrained": False
}

conv4_2_head_group_cfg = {
    "stride": 1,
    "ncpg": 16,
    "pretrained": False
}

mgn_cfg = {
    "num_branches": [1],
    "num_classes": 1,
    "dim": 1
}


multi_branch_classification_cfg = {
    "num_branches": 1,
    "num_classes": 1,
    "local_dim": 1,
    "shared": True
}


multi_task_network_cfg = {
    "num_classes": 1,
    "attributes": {
        "gender": 2,
        "age": 2,
        "backpack": 2
    }
}

pose_cfg = {
    "num_joints": 2,
    "backbone": resnet_groupnorm_cfg
}


pose_reid_cfg = {
    "num_joints": 2,
    "backbone": resnet_groupnorm_cfg,
    "split": False
}


pose_reid_semi_cfg = {
    "num_joints": 2,
    "backbone": resnet_groupnorm_cfg,
    "single_head": True
}


trinet_cfg = {
    "dim": 1
}


def get_cfg(name):
    module = sys.modules[__name__]
    name = name.lower()
    try:
        return getattr(module, '{}_cfg'.format(name))
    except Exception as e:
        raise ValueError("Model config {} not found".format(name))


def build_model(name):
    cfg = get_cfg(name)
    cfg['name'] = name
    cfg['pretrained'] = False
    return model_builder.build(cfg)


@pytest.mark.parametrize("model", get_all_models())
def test_model(model):
    model = build_model(model)
    model.eval()
    test_input = torch.rand(1, 3, 256, 128)
    with torch.no_grad():
        endpoints = model(test_input, model.endpoints)
    print(model.create_endpoints())
    print(model.dimensions)
    # foward one image compare to dimensions
    for key, dim in model.dimensions.items():
        assert endpoints[key].shape[1:] == dim, key
