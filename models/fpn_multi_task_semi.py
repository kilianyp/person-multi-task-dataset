import torch.nn as nn
from .classification import ClassificationBranch
from .attribute import AttributeBranch
from .baseline import BaselineReidBranch
from .pose import Pose2DHead
from .semantic_segmentation import FpnSemHead
from .pose_sem import FpnSemPoseHead
from models import register_model, BaseModel
from builders import model_builder
from models.utils import grad_reverse


@register_model("fpn_head_multi_task_semi")
class FpnHeadMultiTaskNetwork(BaseModel):
    @staticmethod
    def create_endpoints(task_heads):
        endpoints = {}
        for task_head in task_heads.values():
            endpoints.update(task_head.create_endpoints())
        return endpoints

    def __init__(self, backbone, task_heads):
        super().__init__()
        self.task_heads = nn.ModuleDict(task_heads)
        self.endpoints = self.create_endpoints(task_heads)
        self.backbone = backbone

    def forward(self, data, endpoints):
        """Fixed"""
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data)

        for task, task_head in self.task_heads.items():
            if task in ["reid", "attribute", "classification"]:
                endpoints = task_head(reid_x, endpoints)
            elif task == "pose":
                endpoints = task_head(reid_x, endpoints)
                # pose semantic head will otherwise overwrite pose
                endpoints['pose_single'] = endpoints['pose']
            else:
                endpoints = task_head(feature_pyramid, (h, w), endpoints)
                # this should be lip data
                l_pose = endpoints['pose'].detach()
                l_seg = endpoints['sem-logits'].detach()
                endpoints['pose_semi'] = l_pose
                endpoints['sem-logits_semi'] = l_seg


        endpoints["emb"] = endpoints['triplet'][0]
        return endpoints

    @staticmethod
    def build(cfg):
        task_cfgs = cfg['tasks']
        tasks = {}

        for task, task_cfg in task_cfgs.items():
            task_cfg['input_dim'] = 2048
            if task == "attribute":
                tasks[task] = AttributeBranch.build(task_cfg)
            elif task == "classification":
                task_cfg['num_classes'] = cfg['num_classes']
                tasks[task] = ClassificationBranch.build(task_cfg)
            elif task == "pose":
                num_joints = cfg['num_joints']
                tasks[task] = Pose2DHead(2048, num_joints)
            elif task == "reid":
                tasks[task] = BaselineReidBranch.build(task_cfg)
            elif task == "segmentation":
                num_seg_classes = cfg['num_seg_classes']
                tasks[task] = FpnSemHead(num_seg_classes, 256)
            elif task == "pose_segmentation":
                num_seg_classes = cfg['num_seg_classes']
                num_joints = cfg['num_joints']
                tasks[task] = FpnSemPoseHead(num_seg_classes, num_joints)
            else:
                raise ValueError("Unknown task: {}".format(task))

        backbone = model_builder.build(cfg['backbone'])
        return FpnHeadMultiTaskNetwork(backbone, tasks), [], []

