from models import BaseModel, register_model
from builders import model_builder
from models.semantic_segmentation import FpnSemHead
from models.baseline import BaselineReidBranch
from models.utils import grad_reverse
import torch.nn.functional as F
import torch
from torch import nn


@register_model("reid_segmentation")
class ReidSeg(BaseModel):
    def create_endpoints(self):
        endpoints = {}
        endpoints.update(self.sem_head.create_endpoints())
        endpoints["emb"] = None
        endpoints["triplet"] = None
        return endpoints

    def __init__(self, backbone, num_classes, variation):
        super().__init__()
        # expects fpn to return feature pyramid
        self.backbone = backbone
        # fpn has output dim 256
        self.sem_head = FpnSemHead(num_classes, 256)
        self.reid_head = BaselineReidBranch.build({"pooling": "max"})
        self.endpoints = self.create_endpoints()
        print(variation)
        if variation == "v1":
            self.forward = self.forward_v1
        elif variation == "detached":
            self.forward = self.forward_detached
        elif variation == "inversed":
            self.forward = self.forward_inversed
        elif variation == "v2.1":
            self.forward = self.forward_v2_1
        elif variation == "v2.2":
            self.forward = self.forward_v2_2
        elif variation == "v2.3":
            self.forward = self.forward_v2_3
        elif variation == "v2.4":
            self.forward = self.forward_v2_4
            self.attention_linear = nn.Linear(128, 2048)
        else:
            raise ValueError("unknown variation")

    def forward_inversed(self, data, endpoints):
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data, endpoints)
        feature_pyramid = [grad_reverse(x) for x in feature_pyramid]
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
        endpoints = self.reid_head(reid_x, endpoints)
        endpoints['emb'] = endpoints['reid_emb']
        return endpoints

    def forward_detached(self, data, endpoints):
        # explicitly for market reid
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data, endpoints)
        feature_pyramid = [x.detach() for x in feature_pyramid]
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
        endpoints = self.reid_head(reid_x, endpoints)
        endpoints['emb'] = endpoints['reid_emb']

        return endpoints

    def forward_v2_1(self, data, endpoints):
        # with gradient
        # explicitly for market reid
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data, endpoints)
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
        
        sem_logits = endpoints['sem-logits']
        prob = F.softmax(sem_logits)
        background_prob = prob[:, 0]
        # still need channel dim
        background_prob = background_prob.unsqueeze(1)
        foreground_prob = 1 - background_prob
        foreground_prob = F.interpolate(foreground_prob, reid_x.shape[-2:], mode="nearest")
        # weight
        reid_x = foreground_prob * reid_x
        reid_emb = F.adaptive_max_pool2d(reid_x, 1)
        reid_emb = reid_emb.view(reid_emb.shape[0], -1)

        endpoints["triplet"] = [reid_emb]
        endpoints["emb"] = reid_emb

        return endpoints

    def forward_v2_2(self, data, endpoints):
        # without gradient
        # explicitly for market reid
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data, endpoints)
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)

        sem_logits = endpoints['sem-logits'].detach()
        prob = F.softmax(sem_logits)
        background_prob = prob[:, 0]
        # still need channel dim
        background_prob = background_prob.unsqueeze(1)
        foreground_prob = 1 - background_prob
        foreground_prob = F.interpolate(foreground_prob, reid_x.shape[-2:], mode="bilinear")
        # weight
        reid_x = foreground_prob * reid_x
        reid_emb = F.adaptive_max_pool2d(reid_x, 1)
        reid_emb = reid_emb.view(reid_emb.shape[0], -1)

        endpoints["triplet"] = [reid_emb]
        endpoints["emb"] = reid_emb

        return endpoints

    def forward_v2_3(self, data, endpoints):
        # with gradient
        # explicitly for market reid
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data)
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
        
        sem_logits = endpoints['sem-logits']
        prob = F.softmax(sem_logits)
        background_prob = prob[:, 0]
        # still need channel dim
        background_prob = background_prob.unsqueeze(1)
        foreground_prob = 1 - background_prob
        foreground_prob = F.interpolate(foreground_prob, reid_x.shape[-2:], mode="bilinear")
        # weighted
        reid_emb_weighted = foreground_prob * reid_x
        reid_emb_weighted = F.adaptive_max_pool2d(reid_emb_weighted, 1)
        reid_emb_weighted = reid_emb_weighted.view(reid_emb_weighted.shape[0], -1)
        # normal
        reid_emb_normal = F.adaptive_max_pool2d(reid_x, 1)
        reid_emb_normal = reid_emb_normal.view(reid_emb_normal.shape[0], -1)

        reid_emb = torch.cat([reid_emb_normal, reid_emb_weighted], dim=1)

        endpoints["triplet"] = [reid_emb_weighted, reid_emb_normal]
        endpoints["emb"] = reid_emb

        return endpoints

    def forward_v2_4(self, data, endpoints):
        # with gradient
        # explicitly for market reid
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data)
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)

        # normal
        reid_emb = F.adaptive_max_pool2d(reid_x, 1)
        reid_emb = reid_emb.view(reid_emb.shape[0], -1)

        raw_sem  = endpoints['raw-sem']
        raw_sem = F.adaptive_max_pool2d(raw_sem, 1)
        raw_sem = raw_sem.view(raw_sem.shape[0], -1)
        attention_weights = self.attention_linear(raw_sem)

        reid_emb = reid_emb * attention_weights

        endpoints["triplet"] = [reid_emb]
        endpoints["emb"] = reid_emb

        return endpoints

    def forward_v1(self, data, endpoints):
        # fpn also returns first layer before feature pyramid
        # orig_size
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data, endpoints)
        if 'split_info' in data:
            split_info = data['split_info']
            if "lip" in split_info:
                endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
                endpoints["emb"] = None
                endpoints["triplet"] = None
            else:
                endpoints = self.reid_head(reid_x, endpoints)
                endpoints["emb"] = endpoints['reid_emb']
                endpoints["sem-logits"] = None
                endpoints["raw-sem"] = None
        else:
            endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
            endpoints = self.reid_head(reid_x, endpoints)
            endpoints['emb'] = endpoints['reid_emb']

        return endpoints

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        num_classes = cfg['num_seg_classes']
        variation = cfg.get('variation', 'v1')
        model = ReidSeg(backbone, num_classes, variation)
        skips = ["fc"]
        duplicates = []
        return model, skips, duplicates
