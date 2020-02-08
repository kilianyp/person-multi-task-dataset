import torch
import torch.nn as nn
from models import register_model, BaseModel
from builders import model_builder
@register_model('fpn_reid')
class FpnReID(BaseModel):

    @property
    def dimensions(self):
        return {"emb": (self.dim,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['triplet'] = None
        endpoints['emb'] = None
        return endpoints

    def __init__(self, backbone):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__() # 0 classes thows an error
        self.endpoints = self.create_endpoints()
        self.backbone = backbone
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.dim = self.backbone.output_dim * 4
        print(self.dim)

    def forward(self, data, endpoints):
        feature_pyramid, c5 = self.backbone(data)
        triplet_l = []
        for p in feature_pyramid:
            p_emb = self.pooling(p)
            p_emb = p_emb.view(p_emb.shape[0], -1)
            triplet_l.append(p_emb)
        endpoints["emb"] = torch.cat(triplet_l, dim=1)
        endpoints["triplet"] = triplet_l
        return endpoints

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        model = FpnReID(backbone)
        skips = ["fc"]
        return model, skips, []
