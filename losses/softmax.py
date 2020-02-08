import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import get_tensorboard_logger, get_logger
from utils import var2num
from utils import prod


class Sigmoid(nn.Module):
    def __init__(self, target_name='sigmoid', endpoint_name="sigmoid"):
        super().__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss(reduce=False)
        self.tensorboard_logger = get_tensorboard_logger()
        self.logger = get_logger()
        self.endpoint_name = endpoint_name
        self.target_name = target_name

    def forward(self, endpoints, data):
        targets = data[self.target_name]
        acc = calc_acc_sigmoid(endpoints[self.endpoint_name], targets)
        self.tensorboard_logger.add_scalar("acc/sigmoid", acc)
        self.console_logger.add_scalar("acc/sigmoid", acc)
        loss = self.cross_entropy(endpoints[self.endpoint_name], targets)
        self.tensorboard_logger.add_scalar("losses/sigmoid", loss)
        self.logger.info("losses/sigmoid: %f", var2num(loss))
        return loss


def calc_acc_sigmoid(logits, targets, threshold=0.5):
    with torch.no_grad():
        probs = F.sigmoid(logits)
        predicted = (probs > threshold).float()
        return torch.sum(targets == predicted).float() / targets.shape[0]


def calc_acc_softmax(logits, targets):
    with torch.no_grad():
        predicted = torch.max(logits, dim=1)
        predicted = predicted[1]
        return torch.sum(targets == predicted).float() / targets.shape[0]


class Softmax(nn.Module):
    def __init__(self, target_name, endpoint_name):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.tensorboard_logger = get_tensorboard_logger()
        self.logger = get_logger()
        self.target_name = target_name
        self.endpoint_name = endpoint_name

    def forward(self, endpoints, data):
        loss = 0.0
        softmaxs = endpoints[self.endpoint_name]
        targets = data[self.target_name].to(softmaxs[0].device)
        for idx, softmax in enumerate(softmaxs):
            l = self.cross_entropy(softmax, targets)
            loss += l
            self.tensorboard_logger.add_scalar("losses/softmax_{}_{}/min_loss".format(self.endpoint_name, idx), torch.min(l))
            self.tensorboard_logger.add_scalar("losses/softmax_{}_{}/max_loss".format(self.endpoint_name, idx), torch.max(l))
            self.tensorboard_logger.add_scalar("losses/softmax_{}_{}/mean".format(self.endpoint_name, idx), torch.mean(l))

        self.tensorboard_logger.add_scalar("losses/softmax_{}_overall/min_loss".format(self.endpoint_name), torch.min(loss))
        self.tensorboard_logger.add_scalar("losses/softmax_{}_overall/max_loss".format(self.endpoint_name), torch.max(loss))
        self.tensorboard_logger.add_scalar("losses/softmax_{}_overall/mean".format(self.endpoint_name), torch.mean(loss))
        self.logger.info("softmax_%s_overall: %f",
                         self.endpoint_name, var2num(torch.mean(loss)))
        return loss


class CrossEntropyLoss(nn.Module):
    '''
    This class computes the standard crossentropy loss between logits and gt segmentation
    '''
    def __init__(self, target_name, endpoint_name, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        self.tensorboard_logger = get_tensorboard_logger()
        #self.logger = get_logger()
        self.target_name = target_name
        self.endpoint_name = endpoint_name

    def forward(self, endpoints, data):
        logits = endpoints[self.endpoint_name]
        targets = data[self.target_name].to(logits.device)

        loss = self.cross_entropy(logits, targets)
        self.tensorboard_logger.add_scalar("losses/crossentropy_{}".format(self.endpoint_name), loss)
        return loss


def get_topk_percent(tensor, top_k_percent_pixels):
    """
    Returns the top_k pixels of a tensor.
    Similar to
    https://github.com/tensorflow/models/blob/master/research/deeplab/utils/train_utils.py
    Args:
        tensor: At least 2D.
        top_k_percent_pixels (float): percent of pixels we want to return (between 0 and 1)
    """
    assert len(tensor.shape) >= 2
    num_pixels = prod(tensor[0].shape)
    top_k_pixels = int(top_k_percent_pixels * num_pixels)
    tensor = tensor.view(tensor.shape[0], -1)
    return tensor.topk(top_k_pixels)


class BootstrappedCrossEntropyLoss(nn.Module):
    """Use only loss in the top k percent."""
    def __init__(self, target_name, endpoint_name, top_k_percent_pixels=1.0, hard_mining_step=0):
        """
        Args:
            hard_mining_step: Training step in which the hard mining
                kicks off
        """
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

        self.tensorboard_logger = get_tensorboard_logger()
        #self.logger = get_logger()
        self.target_name = target_name
        self.endpoint_name = endpoint_name

        if top_k_percent_pixels == 1.0:
            # just default cross entropy loss
            self.forward = super().forward

        self.top_k_percent_pixels = top_k_percent_pixels
        self.hard_mining_step = hard_mining_step
        # TODO global step
        self.step = 0


    def forward(self, endpoints, data):
        logits = endpoints[self.endpoint_name]
        targets = data[self.target_name].to(logits.device)
        loss = self.cross_entropy(logits, targets)

        if self.hard_mining_step == 0:
            loss, indices = get_topk_percent(loss, self.top_k_percent_pixels)
        else:
            ratio = min(1.0, self.step / self.hard_mining_step)
            top_k_percent_pixels = ratio * self.top_k_percent_pixels + 1 - ratio
            self.tensorboard_logger.add_scalar('losses/topk_pixel', top_k_percent_pixels)
            loss, indices = get_topk_percent(loss, top_k_percent_pixels)
            self.step += 1

        self.tensorboard_logger.add_scalar("losses/bootstrappedce", torch.mean(loss))
        return loss
