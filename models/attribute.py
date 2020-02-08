from torchvision.models.resnet import ResNet, model_urls
from torchvision.models.resnet import Bottleneck
import torch.nn as nn
from models.utils import weights_init_kaiming
from models import register_model, BaseModel


class SingleAttributeBranch(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.fc(x)


class AttributeBranch(nn.Module):
    @property
    def dimensions(self):
        dimensions = {}
        for key, dim in self.attributes.items():
            dimensions[key] = (dim,)
        return dimensions

    def __init__(self, input_dim, attributes, norm="bn"):
        super().__init__()
        if norm == "dropout":
            # probability that it is zeroed
            self.norm_layer = nn.Dropout(p=0.1)
        elif norm == "bn":
            self.norm_layer = nn.BatchNorm1d(input_dim)
            self.norm_layer.apply(weights_init_kaiming)
        elif norm == "gn":
            self.norm_layer = nn.GroupNorm(32, input_dim)
            self.norm_layer.apply(weights_init_kaiming)
        self.attributes = attributes
        self.att_branches = nn.ModuleDict()
        for key, dim in attributes.items():
            self.att_branches['out_' + key] = SingleAttributeBranch(input_dim, dim)
        self.relu = nn.ReLU()
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def create_endpoints(self):
        endpoints = {}
        for key, dim in self.attributes.items():
            endpoints['out_' + key] = None
        endpoints['attribute_emb'] = None
        return endpoints

    def forward(self, x, endpoints):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        attribute_emb = x
        x = self.norm_layer(x)
        x = self.relu(x)
        for key, layer in self.att_branches.items():
            endpoints[key] = layer(x)
        endpoints['attribute_emb'] = attribute_emb
        return endpoints

    @staticmethod
    def build(cfg):
        input_dim = cfg['input_dim']
        attributes = cfg['attributes']
        norm = cfg.get('norm', 'bn')
        return AttributeBranch(input_dim, attributes, norm)


@register_model('attribute')
class Attribute(ResNet, BaseModel):
    """Attribute reid model."""
    model_urls = model_urls
    @property
    def dimensions(self):
        dim = {"emb": (self.inplanes, )}
        dim.update(self.attribute_branch.dimensions)
        return dim

    def create_endpoints(self):
        endpoints = {}
        endpoints['emb'] = None
        endpoints['triplet'] = None
        endpoints.update(self.attribute_branch.create_endpoints())
        return endpoints

    def __init__(self, block, layers, attribute_branch):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.attribute_branch = attribute_branch(self.inplanes)
        self.endpoints = self.create_endpoints()
        self.name = "attribute"
        self.pooling = nn.AdaptiveMaxPool2d(1)

    #TODO: @Killian check signature, does not match resnet
    def forward(self, data, endpoints):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        endpoints = self.attribute_branch(x, endpoints)
        # maybe do nested?
        endpoints["triplet"] = [x]
        endpoints["emb"] = x
        return endpoints

    @staticmethod
    def build(cfg):
        attribute_branch = AttributeBranch.build(cfg)
        model = Attribute(Bottleneck, [3, 4, 6, 3], attribute_branch)
        skips = ['fc']
        return model, skips, []
