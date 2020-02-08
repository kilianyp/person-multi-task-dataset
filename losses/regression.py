from torch import nn
import torch
from logger import get_logger, get_tensorboard_logger
from utils import var2num

class MSELoss(nn.Module):
    """MSE loss.
    Calculates the l2 norm for the matrix target-input.
    """
    def __init__(self, endpoint_name, target_name):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.endpoint_name = endpoint_name
        self.target_name = target_name
        self.logger = get_logger()
        self.tensorboard_logger = get_tensorboard_logger()

    def forward(self, endpoints, data):
        inputs = endpoints[self.endpoint_name]
        targets = data[self.target_name].to(inputs.device)
        loss = self.mse_loss(inputs, targets)
        # mask out nans
        mask = torch.isnan(targets)
        loss[mask] = 0.0
        mask = 1-mask
        num_examples_per_joint = torch.sum(mask, dim=0)
        # NaN * 0 = NaN, set loss to zero before that
        mean_error_per_joint = torch.sum(loss, dim=0) / num_examples_per_joint.float()
        loss = torch.mean(mean_error_per_joint)
        self.logger.info("mse loss %f", var2num(torch.mean(loss)))
        self.tensorboard_logger.add_scalar("losses/mse", torch.mean(loss))
        return loss


class L1Loss(nn.Module):
    def __init__(self, endpoint_name, target_name):
        super().__init__()
        self.endpoint_name = endpoint_name
        self.target_name = target_name
        self.logger = get_logger()
        self.tensorboard_logger = get_tensorboard_logger()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, endpoints, data):
        inputs = endpoints[self.endpoint_name]
        targets = data[self.target_name].to(inputs.device, non_blocking=True)
        loss = self.l1(inputs, targets)
        mask = torch.isnan(targets)
        loss[mask] = 0.0
        mask = 1-mask
        num_examples_per_joint = torch.sum(mask, dim=0)
        # NaN * 0 = NaN, set loss to zero before that
        mean_error_per_joint = torch.sum(loss, dim=0) / num_examples_per_joint.float()
        loss = torch.mean(mean_error_per_joint)
        self.logger.info("l1 loss %f", var2num(loss))
        self.tensorboard_logger.add_scalar("losses/l1", loss)
        return loss


def l2_loss(inputs, targets):
    diff = inputs - targets
    # ignore nan values from start
    loss = torch.sqrt(torch.sum(diff*diff, dim=2) + 1e-12)
    return loss


class L2Loss(nn.Module):
    """ L2 loss
    Calculates the l2 distance between two points
    """
    def __init__(self, endpoint_name, target_name):
        super().__init__()
        self.endpoint_name = endpoint_name
        self.target_name = target_name
        self.logger = get_logger()
        self.tensorboard_logger = get_tensorboard_logger()

    def forward(self, endpoints, data):
        inputs = endpoints[self.endpoint_name]
        # we have to deal with empty inputs (happening because of filtering)
        assert len(inputs.shape) == 3
        targets = data[self.target_name].to(inputs.device, non_blocking=True)
        assert len(targets.shape) == 3
        mask = torch.isnan(targets)
        # loss has to be calculated for all, otherwise we need
        # to later change the loss values which leads to
        # an inplace operation that for some reason
        # cannot propagate the gradient (even when setting to 0)
        targets[mask] = inputs[mask]
        loss = l2_loss(inputs, targets)
        # correctly weigh the loss depending on nan
        mask = 1 - mask[:, :, 0]
        num_joints_per_sample = torch.sum(mask, dim=1)
        non_zero_joints = num_joints_per_sample != 0
        loss_per_sample = torch.sum(loss, dim=1)
        mean_loss_per_sample = loss_per_sample / num_joints_per_sample.float()

        # ignore zero joints
        mean_loss_per_sample = loss_per_sample[non_zero_joints]
        loss = torch.sum(mean_loss_per_sample)/torch.sum(non_zero_joints)
        self.logger.info("l2 loss %f", var2num(torch.mean(loss)))
        self.tensorboard_logger.add_scalar("losses/l2", torch.mean(loss))
        return loss
