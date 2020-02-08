import torch.nn as nn
from models import register_model, BaseModel
from builders import model_builder


def build_pooling_layer(name, output_size=1):
    if name == 'max':
        return nn.AdaptiveMaxPool2d(output_size)
    elif name == 'avg':
        return nn.AdaptiveAvgPool2d(output_size)
    elif name == 'combined':
        return CombinedPooling()
    else:
        raise ValueError


class CombinedPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_pooled = self.max_pooling(x)
        avg_pooled = self.avg_pooling(x)

        return max_pooled + avg_pooled


class BaselineReidBranch(nn.Module):
    @staticmethod
    def create_endpoints():
        return {"triplet": None, 'reid_emb': None}

    def __init__(self, pooling):
        super().__init__()
        self.pooling = pooling

    def forward(self, x, endpoints):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        endpoints['triplet'] = [x]
        endpoints['reid_emb'] = x
        return endpoints

    @staticmethod
    def build(cfg):
        pooling = build_pooling_layer(cfg['pooling'])
        return BaselineReidBranch(pooling)


import torch
class SplitReidBranch(nn.Module):
    @staticmethod
    def create_endpoints():
        return {"triplet": None, 'reid_emb': None}

    def __init__(self, pooling, num_splits):
        super().__init__()
        self.pooling = pooling
        self.num_splits = num_splits
        self.conv = nn.Conv2d(2048, 2048//num_splits, 1)

    def forward(self, x, endpoints):
        x = self.pooling(x)
        x = self.conv(x)
        x = x.view(x.size(0), self.num_splits, -1)
        endpoints['reid_emb'] = x.view(x.size(0), -1)
        x = torch.transpose(x, 0, 1)
        endpoints['triplet'] = x
        return endpoints

    @staticmethod
    def build(cfg):
        num_splits = cfg['num_splits']
        pooling = build_pooling_layer(cfg['pooling'], output_size=(num_splits, 1))
        return SplitReidBranch(pooling, num_splits)


@register_model('baseline')
class Baseline(BaseModel):

    @property
    def dimensions(self):
        return {"emb": (self.dim,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['triplet'] = None
        endpoints['emb'] = None
        return endpoints

    def __init__(self, backbone, reid_branch):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__() # 0 classes thows an error
        self.dim = 2048
        self.endpoints = self.create_endpoints()
        self.reid_branch = reid_branch
        self.backbone = backbone

    def forward(self, x, endpoints):
        x = self.backbone(x, endpoints)
        endpoints = self.reid_branch(x, endpoints)
        endpoints["emb"] = endpoints['reid_emb']
        return endpoints

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        branch_name = cfg.get('branch_type', 'baseline')
        if branch_name == 'baseline':
            reid_branch = BaselineReidBranch.build(cfg)
        elif branch_name == 'split':
            reid_branch = SplitReidBranch.build(cfg)
        else:
            raise ValueError
        model = Baseline(backbone, reid_branch)
        skips = ["fc"]
        return model, skips, []
