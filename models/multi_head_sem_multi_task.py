import torch.nn as nn
from .classification import ClassificationBranch
from .attribute import AttributeBranch
from .baseline import BaselineReidBranch
from .pose import Pose2DHead
from .semantic_segmentation import FpnSemHead
from models import register_model, BaseModel
from builders import model_builder
from models.utils import grad_reverse


@register_model("multi_head_sem_multi_task")
class MultiHeadSemMultiTaskNetwork(BaseModel):
    @staticmethod
    def create_endpoints(task_heads):
        endpoints = {}
        for task_head in task_heads:
            endpoints.update(task_head.create_endpoints())
        endpoints['emb'] = None
        endpoints['sem-logits'] = None
        return endpoints

    def __init__(self, backbone, sem_head, task_heads):
        super().__init__()
        self.task_heads = nn.ModuleList(task_heads)
        self.endpoints = self.create_endpoints(task_heads)
        self.backbone = backbone
        self.sem_head = sem_head
        self.trained_on = {}
        self.trained_on['num_seg_classes'] = sem_head.num_classes

    def forward(self, data, endpoints, **kwargs):
        _, _, h, w = data['img'].size()
        feature_pyramid, task_xs = self.backbone(data, endpoints, **kwargs)
        endpoints = self.sem_head(feature_pyramid, (h, w), endpoints)
        for task_head, task_x in zip(self.task_heads, task_xs):
            endpoints = task_head(task_x, endpoints)
        if 'reid_emb' in endpoints:
            endpoints["emb"] = endpoints['reid_emb']
        return endpoints

    @staticmethod
    def build(cfg):
        task_cfgs = cfg['tasks']
        tasks = []
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
                print("created reid")
                tasks.append(BaselineReidBranch.build(task_cfg))
            else:
                raise ValueError("Unknown task: {}".format(task))

        num_seg_classes = cfg['num_seg_classes']
        sem_head = FpnSemHead(num_seg_classes, 256)
        backbone = model_builder.build(cfg['backbone'])
        model = MultiHeadSemMultiTaskNetwork(backbone, sem_head, tasks)
        return model, [], []

