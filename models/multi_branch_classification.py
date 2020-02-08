from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
import torch.nn.functional as f
import torch
from models import register_model
# minibatch size 128, each identity 4
class SoftmaxBranch(nn.Module):
    def __init__(self, dim, num_classes, conv_1x1):
        super().__init__()
        self.dim = dim
        if isinstance(conv_1x1, nn.Conv2d):
            self.conv_1x1 = conv_1x1
        elif isinstance(conv_1x1, int):
            self.conv_1x1 = nn.Conv2d(conv_1x1, self.dim, 1)
        else:
            raise ValueError("Pass a Conv layer for weight sharing otherwise an int for the Channel group size")

        self.batchnorm = nn.BatchNorm1d(self.dim)
        self.linear = nn.Linear(self.dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1x1(x)
        x = x.view(x.size(0), -1)
        emb = x
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.linear(x)
        return x, emb


@register_model("multi_branch_classification")
class MultiBranchClassification(ResNet):
    @property
    def dimensions(self):
        return {"emb": (self.local_dim * self.Nc,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['softmax'] = None
        endpoints['emb'] = None
        return endpoints

    def __init__(self, block, layers, num_classes, num_branches, local_dim, shared=True,**kwargs):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1) # 0 classes thows an error
        self.name = "MultiBranchClassification"
        # resnet output dim
        self.global_dim = 2048
        self.local_dim = local_dim
        self.shared = shared
        self.Nc = num_branches
        self.dim = self.local_dim * self.Nc # embedding_dim
        self.num_classes = num_classes

        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1) 
        if self.global_dim % self.Nc != 0:
            raise RuntimeError("Please ensure that the global dimension {} is dividable"
                    "by the number of channel groups {}".format(self.global_dim, self.Nc))
        self.Cg = self.global_dim // self.Nc

        if shared:
            self.conv_1x1 = nn.Conv2d(self.Cg, self.local_dim, 1)
        else:
            self.conv_1x1 = self.Cg

        self.branches = nn.ModuleList()

        for _ in range(self.Nc):
            self.branches.append(SoftmaxBranch(self.local_dim,
                num_classes, self.conv_1x1))

        self.endpoints = self.create_endpoints()


    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = f.avg_pool2d(x, x.size()[2:])
        softmaxs = []
        embs = []
        for i in range(self.Nc):
            softmax, emb = self.branches[i](x[:, i*self.Cg : (i+1)*self.Cg])
            softmaxs.append(softmax)
            embs.append(emb)

        endpoints["softmax"] = softmaxs
        endpoints["emb"] = torch.cat(embs, dim=1)
        return endpoints

    @staticmethod
    def build(cfg):
        num_classes = cfg['num_classes']
        num_branches = cfg['num_branches']
        local_dim = cfg['local_dim']
        shared = cfg['shared']
        model = MultiBranchClassification(Bottleneck, [3, 4, 6, 3], num_classes,
                                          num_branches, local_dim, shared)
        # filter out fully connected keys
        skips = ['fc']

        return model, skips, []

