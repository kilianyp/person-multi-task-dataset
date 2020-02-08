import torch.nn as nn
from .classification import ClassificationBranch
from .attribute import AttributeBranch
from .baseline import BaselineReidBranch
from .pose import Pose2DHead
from models import register_model, BaseModel
from builders import model_builder
from models.utils import grad_reverse


@register_model("multi_head_reid_multi_task")
class MultiHeadMultiTaskNetwork(BaseModel):
    @staticmethod
    def create_endpoints(task_heads):
        endpoints = {}
        for task_head in task_heads:
            endpoints.update(task_head.create_endpoints())
        return endpoints

    def __init__(self, backbone, reid_head, task_heads):
        super().__init__()
        self.task_heads = nn.ModuleList(task_heads)
        self.endpoints = self.create_endpoints(task_heads)
        self.backbone = backbone
        self.reid_head = reid_head

    def forward(self, x, endpoints, **kwargs):
        heads_x = self.backbone(x, endpoints, **kwargs)
        reid_x = heads_x[0]
        task_x = heads_x[1:]
        endpoints = self.reid_head(reid_x, endpoints)
        endpoints['emb'] = endpoints['reid_emb']
        # do all on top of embedding
        for idx, task_head in enumerate(self.task_heads):
            endpoints = task_head(task_x[idx], endpoints)
        return endpoints

    @staticmethod
    def build(cfg):
        task_cfgs = cfg['tasks']
        tasks = []
        mode = cfg["mode"]
        if mode not in ['inversed', 'detached', 'normal']:
            raise ValueError(mode)

        for task, task_cfg in task_cfgs.items():
            # depending on the backbone model
            task_cfg['input_dim'] = 2048
            if task == "attribute":
                tasks.append(AttributeBranch.build(task_cfg))
            elif task == "classification":
                task_cfg['num_classes'] = cfg['num_classes']
                tasks.append(ClassificationBranch.build(task_cfg))
            elif task == "pose":
                task_cfg['num_joints'] = cfg['num_joints']
                tasks.append(Pose2DHead.build(task_cfg))
            elif task == 'reid':
                tasks.append(BaselineReidBranch.build(task_cfg))
            else:
                raise ValueError("Unknown task: {}".format(task))

        reid_head = BaselineReidBranch.build(cfg['reid'])
        backbone = model_builder.build(cfg['backbone'])
        return MultiHeadMultiTaskNetwork(backbone, reid_head, tasks), [], []

