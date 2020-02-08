from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck
import torch.nn as nn
from models import register_model

def make_layer(block, planes, blocks, stride=1):
    """Copied from torchvision/models/resnet.py
    Adapted to always be follow after layer3
    """
    # layer3 has 256 * block.expansion output channels
    inplanes = 256 * block.expansion #here
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    for i in range(1, blocks):
        layers.append(block(planes * block.expansion, planes)) #here

    return nn.Sequential(*layers)


class TaskBranch(nn.Module):
    def __init__(self, block, layers, task):
        super().__init__()
        self.task = task
        # set inplanes for block building
        self.inplanes = 256 * block.expansion
        self.layer4 = make_layer(block, 512, layers, stride=1)

    def forward(self, x, endpoints):
        x = self.layer4(x)
        endpoints = self.task(x, endpoints)
        return endpoints


@register_model("conv4_multi_task")
class Conv4MultiTask(ResNet):
    """Archtitecture that creates separate branches
    after layer3 ('conv4_x')
    """
    @property
    def dimensions(self):
        dimensions = {'emb': (2048,)}
        for task in self.tasks:
            dimensions.update(task.dimensions)
        return dimensions

    def create_endpoints(self):
        endpoints = {'emb': None}
        for task in self.tasks:
            endpoints.update(task.create_endpoints())
        return endpoints

    def __init__(self, block, layers, tasks):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.tasks = tasks
        self.task_branches = nn.ModuleList()
        for task in tasks:
            self.task_branches.append(TaskBranch(block, layers[3], task))
        self.endpoints = self.create_endpoints()
        self.fc = None

    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        for task_branch in self.task_branches:
            endpoints, emb = task_branch(x, endpoints)

        emb = endpoints['triplet'][0]
        endpoints['emb'] = emb
        # merging module
        return endpoints

    @staticmethod
    def build(cfg):
        task_cfgs = cfg['tasks']
        tasks = []
        for type, task_cfg in task_cfgs.items():
            tasks.append(build_task_branch(type, task_cfg))
        model = Conv4MultiTask(Bottleneck, [3, 4, 6, 3], tasks)
        duplicate = []
        to_layers4 = []
        for idx in range(len(model.tasks)):
            to_layers4.append("task_branches.{}.layer4.".format(idx))
        duplicate.append(("layer4", to_layers4))

        # TODO sometimes we need to skip them, sometimes we do not?
        skips = ["fc", "layer4"]
        return model, skips, duplicate


def build_task_branch(type, cfg):
    if type == 'reid':
        from models.baseline import BaselineReidBranch
        return BaselineReidBranch.build(cfg)
    elif type == 'pose':
        from models.pose import Pose2DHead
        return Pose2DHead.build(cfg)
    else:
        raise ValueError
