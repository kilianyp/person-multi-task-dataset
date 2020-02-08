from resnet_groupnorm.resnet import ResNet as ResNetGroup, Bottleneck as BottleneckGroup
from models import register_model, BaseModel
from torch import nn


@register_model("conv4_multi_head_group")
class Conv4MultiHeadGroup(ResNetGroup, BaseModel):
    def __init__(self, block, layers, stride, ncpg, num_heads):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1, ncpg)
        self.layer4 = nn.ModuleList()
        self.num_heads = num_heads
        for _ in range(num_heads):
            self.inplanes = 256 * block.expansion
            layer4 = self._make_layer(block, 512, layers[3], stride=stride, group_norm=ncpg)
            self.layer4.append(layer4)
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
        head_x = []
        for layer4 in self.layer4:
            head_x.append(layer4(x))
        return head_x

    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        ncpg = cfg['ncpg']
        num_heads = cfg['num_heads']
        model = Conv4MultiHeadGroup(BottleneckGroup, [3, 4, 6, 3], stride, ncpg, num_heads)
        duplicate = []
        layer4_duplicates = []
        for idx in range(num_heads):
            layer4_duplicates.append("layer4.{}".format(idx))
        duplicate.append(("layer4", layer4_duplicates))
        skips = ["fc"]
        return model, skips, duplicate


from torchvision.models.resnet import ResNet,  Bottleneck, model_urls
@register_model("conv4_multi_head_batch")
class Conv4MultiHead(ResNet, BaseModel):
    model_urls = model_urls

    def __init__(self, block, layers, stride, num_heads):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.num_heads = num_heads
        self.layer4 = nn.ModuleList()
        for _ in range(num_heads):
            self.inplanes = 256 * block.expansion
            layer4 = self._make_layer(block, 512, layers[3], stride=stride)
            self.layer4.append(layer4)
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
        head_x = []
        for layer4 in self.layer4:
            head_x.append(layer4(x))
        return head_x

    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        num_heads = cfg['num_heads']
        model = Conv4MultiHead(Bottleneck, [3, 4, 6, 3], stride, num_heads)
        duplicate = []
        layer4_duplicates = []
        for idx in range(num_heads):
            layer4_duplicates.append("layer4.{}".format(idx))
        duplicate.append(("layer4", layer4_duplicates))
        skips = ["fc"]
        return model, skips, duplicate
