import torch
import torch.nn.functional as F
import torch.nn as nn
from models import register_model, BaseModel
from builders import model_builder


class SoftArgMax2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_spacing = None
        self.y_spacing = None

    def forward(self, x):
        """
        Takes input and calculates a value from 0 to 1 along x and 0 to 1 along y direction
        for each joint
        args:
            x (Tensor): A tensor of shape (batch_size, width, height, num_joints)
                        where depth is the dimension of the point.
        """
        batch_size, num_joints, height, width = x.shape

        if self.x_spacing is None:
            self.x_spacing = torch.linspace(0, 1, width).to(x.device)
            self.y_spacing = torch.linspace(0, 1, height).to(x.device)
            self.width = width
            self.height = height

        if self.height != height:
            self.y_spacing = torch.linspace(0, 1, height).to(x.device)
            self.height = height

        if self.width != width:
            self.x_spacing = torch.linspace(0, 1, width).to(x.device)
            self.width = width

        x = x.reshape(batch_size, num_joints, -1)
        x = F.softmax(x, dim=2)
        x = x.reshape(batch_size, num_joints, height, width)
        x_coord = (x.sum(dim=2) * self.x_spacing).sum(dim=2)
        y_coord = (x.sum(dim=3) * self.y_spacing).sum(dim=2)
        return torch.stack([x_coord, y_coord], dim=2)


class Pose2DHead(nn.Module):
    """A very simple 2D output head.
    """

    @staticmethod
    def create_endpoints():
        return {'pose': None}

    @property
    def dimensions(self):
        return {"pose": (self.num_joints, 2)}

    def __init__(self, in_channels, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        # for each coordinate
        self.conv1 = nn.Conv2d(in_channels, num_joints, 1)
        # softargmax and loss is not joint
        self.softargmax = SoftArgMax2d()

    def forward(self, x, endpoints):
        # emb = F.max_pool2d(x, x.size()[2:])
        # emb = emb.view(x.size(0), -1)

        # expects already batchnormed input
        x = self.conv1(x)
        endpoints['pose'] = self.softargmax(x)
        return endpoints

    @staticmethod
    def build(cfg):
        input_dim = cfg['input_dim']
        return Pose2DHead(input_dim, cfg['num_joints'])


@register_model("pose")
class Pose(BaseModel):
    @property
    def dimensions(self):
        return {"emb": (self.dim,), "pose": (self.num_joints, 2)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['emb'] = None
        endpoints['pose'] = None
        endpoints['pose_maps'] = None
        return endpoints

    def __init__(self, backbone, num_joints):
        super().__init__()
        self.backbone = backbone
        self.endpoints = self.create_endpoints()
        self.dim = self.backbone.inplanes
        self.num_joints = num_joints
        self.posehead = Pose2DHead(self.dim, num_joints)

    def forward(self, x, endpoints):
        x = self.backbone(x)
        endpoints = self.posehead(x, endpoints)
        endpoints['emb'] = None
        return endpoints

    @staticmethod
    def build(cfg):
        num_joints = cfg['num_joints']
        backbone = model_builder.build(cfg['backbone'])

        model = Pose(backbone, num_joints)
        skips = []
        duplicate = []

        return model, skips, duplicate
