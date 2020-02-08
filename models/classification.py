from builders import model_builder
import torch.nn as nn
from models.utils import weights_init_classifier, weights_init_kaiming
from models import register_model, BaseModel
import builders.merging_block_builder as merging_block_builder
from resnet_groupnorm.group_norm import GroupNorm1D


class ClassificationBranch(nn.Module):
    def __init__(self, input_dim, num_classes, with_relu=True, norm="bn"):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes, bias=False)
        self.linear.apply(weights_init_classifier)
        if norm == "bn":
            self.norm_layer = nn.BatchNorm1d(input_dim)
            self.norm_layer.bias.requires_grad_(False) # no shift
        elif norm == "gn":
            self.norm_layer = nn.GroupNorm(32, input_dim)
        elif norm == "gn_v2":
            self.norm_layer = GroupNorm1D(input_dim, 16, affine=True)
        self.norm_layer.apply(weights_init_kaiming)

        self.with_relu = with_relu
        if with_relu:
            self.relu = nn.ReLU()
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x, endpoints):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        if self.with_relu:
            x = self.relu(x)
        x = self.norm_layer(x)
        emb = x
        x = self.linear(x)
        endpoints['softmax'] = [x]
        endpoints['class_emb'] = emb
        return endpoints

    @staticmethod
    def create_endpoints():
        return {'softmax': None}

    @staticmethod
    def build(cfg):
        input_dim = cfg['input_dim']
        num_classes = cfg['num_classes']
        with_relu = cfg.get('with_relu', True)
        norm = cfg.get('norm', "bn")
        return ClassificationBranch(input_dim, num_classes, with_relu, norm)


@register_model('classification')
class Classification(BaseModel):
    @property
    def dimensions(self):
        return {"emb": (self.emb_dim,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['emb'] = None
        endpoints['softmax'] = None
        return endpoints

    def __init__(self, backbone, class_branch, num_classes):
        """Initializes original ResNet and overwrites fully connected layer."""
        super().__init__()  # 0 classes thows an error
        self.fc = None
        # reset inplanes
        self.num_classes = num_classes
        self.backbone = backbone
        self.classification = class_branch
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.endpoints = self.create_endpoints()
        self.name = "classification"

    def forward(self, data, endpoints):
        x = self.backbone(data, endpoints)
        endpoints = self.classification(x, endpoints)
        endpoints['emb'] = endpoints['class_emb']
        # if not self.training:
        #    endpoints["emb"] = self.merging_block(endpoints)
        return endpoints

    @staticmethod
    def build(cfg):
        num_classes = cfg['num_classes']
        merging_block = merging_block_builder.build(cfg.get('merging_block'))
        backbone = model_builder.build(cfg['backbone'])
        cfg['input_dim'] = 2048
        class_branch = ClassificationBranch.build(cfg)
        model = Classification(backbone, class_branch, num_classes)
        skips = ['fc']
        duplicate = []
        return model, skips, duplicate
