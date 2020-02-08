import numpy as np
from models.pose import SoftArgMax2d
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, channels, num_joints):
        super().__init__()
        self.conv = nn.Conv2d(channels, num_joints, 1)

    def forward(self, x):
        return self.conv(x)


def test_softargmax():
    batch_size = 2
    height = 16
    width = 12
    num_joints = 5
    channels = 2048
    model = Model(channels, num_joints)
    soft = SoftArgMax2d()
    x = torch.rand(batch_size, channels, height, width)
    targets = torch.rand(batch_size, num_joints, 2, requires_grad=False)
    targets[0, 0] = np.nan
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)
    mse = nn.MSELoss(reduction='none')
    mse = nn.L1Loss(reduction='none')
    print(targets)
    for i in range(1000):
        inputs = model(x)
        inputs = soft(inputs)
        loss = mse(inputs, targets)
        if i == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        mask = torch.isnan(targets)
        loss[mask] = 0.0
        mask = 1-mask
        num_examples_per_joint = torch.sum(mask, dim=0)
        # NaN * 0 = NaN, set loss to zero before that
        mean_error_per_joint = torch.sum(loss, dim=0) / num_examples_per_joint.float()
        loss = torch.mean(mean_error_per_joint)
        loss.backward()
        optimizer.step()
    print(torch.mean(loss))
    print(inputs)
