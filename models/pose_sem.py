from models import BaseModel, register_model
from models.semantic_segmentation import FpnSemHead
from models.pose import Pose2DHead
from builders import model_builder
import torch.nn.functional as F


class FpnSemPoseHead(FpnSemHead):
    def __init__(self, num_classes, num_joints, input_dim=256, output_dim=128):
        super().__init__(num_classes, input_dim=input_dim, output_dim=output_dim)
        # output dim of fpn pyramid is by default 128
        self.pose_head = Pose2DHead(output_dim, num_joints)

    def forward(self, feature_pyramid, orig_size, endpoints):
        p2, p3, p4, p5 = feature_pyramid
        f5 = self.p5upsample(p5)
        f5 = self.upsample_to_fixed_size(f5, p2)
        f4 = self.p4upsample(p4)
        f4 = self.upsample_to_fixed_size(f4, p2)
        f3 = self.p3upsample(p3)
        f3 = self.upsample_to_fixed_size(f3, p2)
        f2 = self.p2upsample(p2)


        f1 = f2 + f3 + f4 + f5
        endpoints = self.pose_head(f1, endpoints)
        # to num_seg_classe
        f1 = self.conv1(f1)
        logits = F.upsample(f1, size=orig_size, mode='bilinear')
        endpoints['sem-logits'] = logits

        return endpoints


@register_model("pose_semantic")
class PoseSemantic(BaseModel):
    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['pose'] = None
        endpoints['sem-logits'] = None
        return endpoints

    def __init__(self, backbone, num_seg_classes, num_joints):
        super().__init__()
        # expects fpn to return feature pyramid
        self.backbone = backbone
        self.dim = self.backbone.inplanes
        self.pose_sem_head = FpnSemPoseHead(num_seg_classes, num_joints)
        self.endpoints = self.create_endpoints()

    def forward(self, data, endpoints):
        # fpn also returns first layer before feature pyramid
        feature_pyramid, layer5 = self.backbone(data)
        _, _, h, w = data['img'].size()
        endpoints, seg_emb = self.pose_sem_head(feature_pyramid, (h, w), endpoints)
        return endpoints

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        num_seg_classes = cfg['num_seg_classes']
        num_joints = cfg['num_joints']
        model = PoseSemantic(backbone, num_seg_classes, num_joints)

        skips = ["fc"]
        duplicates = []
        return model, skips, duplicates

