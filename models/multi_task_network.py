from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
from .classification import ClassificationBranch
from .attribute import AttributeBranch
from models.utils import weights_init_kaiming
from models import register_model


class MultiTaskNetwork(ResNet):
    @property
    def dimensions(self):
        return self._dimensions

    @staticmethod
    def create_endpoints(attributes):
        endpoints = {}
        endpoints['emb'] = None
        endpoints['triplet'] = None
        endpoints['softmax'] = None
        for key, dim in attributes.items():
            endpoints[key] = None
        return endpoints

    def __init__(self, block, layers, num_classes, attributes):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.dim = 2048
        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.batchnorm = nn.BatchNorm1d(self.inplanes)
        self.batchnorm.apply(weights_init_kaiming)

        self.emb_dim = 2048

        self.classification = ClassificationBranch(self.inplanes, num_classes)
        self.num_classes = num_classes

        self.endpoints = self.create_endpoints(attributes)

        self.attributes = attributes
        self.att_branches = nn.ModuleDict()

        for key, dim in attributes.items():
            self.att_branches[key] = AttributeBranch(dim)

        self._dimensions = {"emb": (2048,)}
        for key, dim in attributes.items():
            self._dimensions[key] = (dim,)

    def forward(self, x, endpoints):
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

        # classification
        softmax, softmax_emb = self.classification(x)

        # attribute
        normed = self.batchnorm(x)
        for key, layer in self.att_branches.items():
            endpoints[key] = layer(normed)
        endpoints["softmax"] = [softmax]
        # triplet
        endpoints["triplet"] = [x]
        endpoints["emb"] = x
        return endpoints

    @staticmethod
    def build(cfg):
        num_classes = cfg['num_classes']
        attributes = cfg['attributes']

        model = MultiTaskNetwork(Bottleneck, [3, 4, 6, 3],
                                 num_classes, attributes)

        skips = ['fc']
        return model, skips, []
