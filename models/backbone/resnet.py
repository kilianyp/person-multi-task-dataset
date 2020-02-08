from torchvision.models.resnet import ResNet, Bottleneck, model_urls
from models import register_model, BaseModel
from torch import nn

@register_model("ResNet")
class BatchnormBackbone(ResNet, BaseModel):
    model_urls = model_urls
    def __init__(self, block, layers, stride=1):
        super().__init__(block, layers, 1)
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride)
        self.fc = None

    def forward(self, data, endpoints):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        model = BatchnormBackbone(Bottleneck, [3, 4, 6, 3], stride)
        skips = ['fc']
        return model, skips, []

from resnet_groupnorm.resnet import ResNet as ResNetGroup, Bottleneck as BottleneckGroup
@register_model("ResNet_Groupnorm")
class GroupnormBackbone(ResNetGroup, BaseModel):
    def __init__(self, block, layers, ncpg, stride):
        """Initializes groupnorm ResNet and overwrites fully
        connected layer.

        Args:
            ncpg: number channels per group. ncpg=0 corresponds toi
                  normal BatchNorm2d
        """
        # 0 classes thows an error
        super().__init__(block, layers, 1, ncpg)
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride, group_norm=ncpg)
        self.fc = None
        # remove last relu
#        self.layer4[-1].relu = nn.Sequential()

    def forward(self, data, endpoints):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        # backwards compability
        if 'ncpg' in cfg:
            ncpg = cfg['ncpg']
        else:
            ncpg = cfg['num_groups']

        model = GroupnormBackbone(BottleneckGroup, [3, 4, 6, 3], ncpg, stride)
        skips = ['fc']
        return model, skips, []
