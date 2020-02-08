from torch import nn
from .pose import Pose2DHead, SoftArgMax2d
from models import register_model, BaseModel
from builders import model_builder


@register_model("pose_reid")
class PoseReid(BaseModel):
    @property
    def dimensions(self):
        return {"emb": (self.dim,), "pose": (self.num_joints, 2)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['emb'] = None
        endpoints['pose'] = None
        endpoints['triplet'] = None
        return endpoints

    def __init__(self, backbone, split, num_joints, single_head):
        super().__init__()

        if not single_head and split:
            raise RuntimeError("Dual head and splitting is an invalid configuration.")

        self.backbone = backbone
        self.inplanes = backbone.inplanes
        self.endpoints = self.create_endpoints()
        self.num_joints = num_joints
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.single_head = single_head
        self.split = split


        if self.split:
            self.dim = self.inplanes - self.num_joints
            self.softargmax = SoftArgMax2d()
            self._forward_fn = self._forward_single_head_split

        else:
            if single_head:
                self._forward_fn = self._forward_single_head_not_split
            else:
                self._forward_fn = self._forward_dual_head

            self.dim = self.inplanes
            self.pose_head = Pose2DHead(self.inplanes, self.num_joints)

    def _forward_single_head_split(self, x, endpoints):
        x = self.backbone(x, endpoints)
        pose_data = x[:, -self.num_joints:, :, :]
        #print(pose_data.shape)
        endpoints['pose'] = self.softargmax(pose_data)

        triplet_data = x[:, :-self.num_joints, :, :]
        #print(triplet_data.shape)
        triplet_data = self.pooling(triplet_data)
        triplet_data = triplet_data.view(triplet_data.size(0), -1)
        endpoints['emb'] = triplet_data
        endpoints['triplet'] = [triplet_data]
        return endpoints

    def _forward_single_head_not_split(self, x, endpoints):
        x = self.backbone(x, endpoints)
        endpoints = self.pose_head(x, endpoints)
        r = self.pooling(x)
        r = r.view(r.size(0), -1)
        endpoints['emb'] = r
        endpoints['triplet'] = [r]
        return endpoints

    def _forward_dual_head(self, x, endpoints):
        p, r = self.backbone(x, endpoints)
        endpoints = self.pose_head(p, endpoints)
        r = self.pooling(r)
        r = r.view(r.size(0), -1)
        endpoints['emb'] = r
        endpoints['triplet'] = [r]
        return endpoints

    def forward(self, x, endpoints):
        return self._forward_fn(x, endpoints)

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        num_joints = cfg['num_joints']
        split = cfg['split']
        single_head = cfg['single_head']
        model = PoseReid(backbone, split, num_joints, single_head)
        skips = []
        return model, skips, []
