import torch.nn as nn
from .classification import ClassificationBranch
from .attribute import AttributeBranch
from .baseline import BaselineReidBranch
from .pose import Pose2DHead
from .semantic_segmentation import FpnSemHead
from models import register_model, BaseModel
from builders import model_builder
from models.utils import grad_reverse


@register_model("single_head_reid_sem_multi_task")
class SingleHeadMultiTaskNetwork(BaseModel):
    @staticmethod
    def create_endpoints(task_heads):
        endpoints = {}
        for task_head in task_heads:
            endpoints.update(task_head.create_endpoints())
        endpoints['emb'] = None
        endpoints['sem-logits'] = None
        return endpoints

    def __init__(self, backbone, reid_head, sem_head, task_heads):
        super().__init__()
        self.task_heads = nn.ModuleList(task_heads)
        self.endpoints = self.create_endpoints(task_heads)
        self.backbone = backbone
        self.reid_head = reid_head
        self.sem_head = sem_head

    def forward(self, data, endpoints, **kwargs):
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data, endpoints, **kwargs)
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
        endpoints =  self.reid_head(reid_x, endpoints)

        for task_head in self.task_heads:
            endpoints = task_head(reid_x, endpoints)
        endpoints["emb"] = endpoints['reid_emb']
        return endpoints

    @staticmethod
    def build(cfg):
        task_cfgs = cfg['tasks']
        tasks = []

        for task, task_cfg in task_cfgs.items():
            task_cfg['input_dim'] = 2048
            if task == "attribute":
                tasks.append(AttributeBranch.build(task_cfg))
            elif task == "classification":
                task_cfg['num_classes'] = cfg['num_classes']
                tasks.append(ClassificationBranch.build(task_cfg))
            elif task == "pose":
                num_joints = cfg['num_joints']
                tasks.append(Pose2DHead(2048, num_joints))
            else:
                raise ValueError("Unknown task: {}".format(task))

        reid_head = BaselineReidBranch.build(cfg['reid'])
        num_seg_classes = cfg['num_seg_classes']
        sem_head = FpnSemHead(num_seg_classes, 256)
        backbone = model_builder.build(cfg['backbone'])
        return SingleHeadMultiTaskNetwork(backbone, reid_head, sem_head, tasks), [], []

