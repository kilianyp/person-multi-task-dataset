from resnet_groupnorm.resnet import ResNet as ResNetGroup, Bottleneck as BottleneckGroup
from models import register_model, BaseModel


@register_model("conv4_2_head_group")
class Conv4TwoHeadGroup(ResNetGroup, BaseModel):
    def __init__(self, block, layers, stride, ncpg):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1, ncpg)
        self.inplanes = 256 * block.expansion
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=stride, group_norm=ncpg)
        self.inplanes = 256 * block.expansion
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=stride, group_norm=ncpg)
        self.fc = None

    def forward(self, data):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        layer4_1 = self.layer4_1(x)
        layer4_2 = self.layer4_2(x)
        return layer4_1, layer4_2


    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        ncpg = cfg['ncpg']
        model = Conv4TwoHeadGroup(BottleneckGroup, [3, 4, 6, 3], stride, ncpg)
        duplicate = []
        duplicate.append(("layer4", ["layer4_1", "layer4_2"]))
        skips = ["fc"]
        return model, skips, duplicate


from torchvision.models.resnet import ResNet,  Bottleneck
@register_model("conv4_2_head_batch")
class Conv4TwoHead(ResNet, BaseModel):

    def __init__(self, block, layers, stride):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.inplanes = 256 * block.expansion
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=stride)
        self.inplanes = 256 * block.expansion
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=stride)
        self.fc = None

    def forward(self, data):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        layer4_1 = self.layer4_1(x)
        layer4_2 = self.layer4_2(x)
        return layer4_1, layer4_2


    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        model = Conv4TwoHead(Bottleneck, [3, 4, 6, 3], stride)
        duplicate = []
        duplicate.append(("layer4", ["layer4_1", "layer4_2"]))
        skips = ["fc"]
        return model, skips, duplicate

