from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
from .classification import ClassificationBranch
from models import register_model

@register_model("classification_triplet")
class ClassificationTriplet(ResNet):

    @property
    def dimensions(self):
        return {'emb': (self.emb_dim,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['triplet'] = None
        endpoints['emb'] = None
        endpoints['softmax'] = None
        return endpoints

    def __init__(self, block, layers, num_classes, merging_block=None):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.dim = 2048
        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.endpoints = self.create_endpoints()
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.classification = ClassificationBranch(self.inplanes, num_classes)
        self.num_classes = num_classes
        self.emb_dim = 2048
        self.merging_block = merging_block

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
        softmax, softmax_emb = self.classification(x)
        endpoints["softmax"] = [softmax]
        endpoints["triplet"] = [x]
        endpoints["emb"] = x
        return endpoints

    @staticmethod
    def build(cfg):
        num_classes = cfg['num_classes']
        model = ClassificationTriplet(Bottleneck, [3, 4, 6, 3], num_classes)
        skips = ['fc']
        duplicate = []
        return model, skips, duplicate

