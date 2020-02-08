import torch.nn as nn
from models import register_model, BaseModel
from builders import model_builder

@register_model('trinet')
class TriNet(BaseModel):
    """TriNet implementation.

    Replaces the last layer of ResNet50 with two fully connected layers.

    First: 1024 units with batch normalization and ReLU
    Second: 128 units, final embedding.
    """

    def __init__(self, backbone, dim=128):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__() # 0 classes thows an error
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(2048, dim)
        self.backbone = backbone
        self.dim = dim
        self.endpoints = self.create_endpoints()


    def forward(self, x, endpoints):
        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        endpoints["emb"] = x
        endpoints["triplet"] = [x]
        return endpoints

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints["emb"] = None
        endpoints["triplet"] = None
        return endpoints


    @staticmethod
    def build(cfg):
        dim = cfg['dim']
        backbone = model_builder.build(cfg['backbone'])
        model = TriNet(backbone, dim)
        skips = ['fc']
        return model, skips, []

