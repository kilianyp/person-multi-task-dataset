"""
Adapted from https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
"""
from torch import nn
from models import register_model
from models.backbone.resnet import GroupnormBackbone, BottleneckGroup
import torch.nn.functional as F


@register_model("fpn_two_head")
class Fpn(GroupnormBackbone):
    def __init__(self, block, layers, ncpg, stride, output_dim=256):
        super().__init__(block, layers, ncpg, stride)
        self.output_dim = output_dim
        self.num_stages = 4  # pyramid stages

        self.inplanes = 256 * block.expansion
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=1, group_norm=ncpg)
        # Top layer
        self.toplayer = nn.Conv2d(2048, output_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, output_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, output_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, output_dim, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''
        Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, data, endpoints):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c5_2 = self.layer4_2(c4)

        #TODO: Possibly compute avg pooling and add it back
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return (p2, p3, p4, p5), c5_2

    @staticmethod
    def build(cfg):
        stride = cfg['stride']
        # backwards compability
        if 'ncpg' in cfg:
            ncpg = cfg['ncpg']
        else:
            ncpg = cfg['num_groups']
        output_dim = cfg['output_dim']
        model = Fpn(BottleneckGroup, [3, 4, 6, 3], ncpg, stride, output_dim)
        duplicate = [("layer4", ["layer4_2"])]
        skips = ['fc']
        return model, skips, duplicate
