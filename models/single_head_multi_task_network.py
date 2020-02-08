import torch.nn as nn
from .classification import ClassificationBranch
from .attribute import AttributeBranch
from .baseline import BaselineReidBranch
from .pose import Pose2DHead
from models import register_model, BaseModel
from builders import model_builder
from models.utils import grad_reverse


@register_model("single_head_reid_multi_task")
class SingleHeadMultiTaskNetwork(BaseModel):
    @staticmethod
    def create_endpoints(task_heads):
        endpoints = {}
        for task_head in task_heads:
            endpoints.update(task_head.create_endpoints())
        return endpoints

    def __init__(self, backbone, reid_head, task_heads, mode):
        super().__init__()
        self.task_heads = nn.ModuleList(task_heads)
        self.endpoints = self.create_endpoints(task_heads)
        self.backbone = backbone
        self.reid_head = reid_head
        self.mode = mode

    def forward(self, x, endpoints, **kwargs):
        x = self.backbone(x, endpoints, **kwargs)
        endpoints = self.reid_head(x, endpoints)
        endpoints['emb'] = endpoints['reid_emb']
        # do all on top of embedding
        if self.mode == 'detached':
            x = x.detach()
        elif self.mode == 'inversed':
            x = grad_reverse(x)

        for task_head in self.task_heads:
            endpoints = task_head(x, endpoints)
        return endpoints

    @staticmethod
    def build(cfg):
        task_cfgs = cfg['tasks']
        tasks = []
        mode = cfg["mode"]
        if mode not in ['inversed', 'detached', 'normal']:
            raise ValueError(mode)

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
        backbone = model_builder.build(cfg['backbone'])
        return SingleHeadMultiTaskNetwork(backbone, reid_head, tasks, mode), [], []

