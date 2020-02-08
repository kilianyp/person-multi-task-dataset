import torch
import torch.nn as nn
from logger import get_tensorboard_logger
class Attribute(nn.Module):
    def __init__(self, attributes):
        super().__init__()
        self.attributes = attributes
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.M = len(attributes)
        self.tensorboard_logger = get_tensorboard_logger()

    def forward(self, endpoints, data):
        loss = 0.0
        for attribute in self.attributes:
            logits = endpoints['out_' + attribute]
            targets = data[attribute].to(logits[0].device, non_blocking=True)
            l = self.cross_entropy(logits, targets)
            loss += l
            self.tensorboard_logger.add_scalar("losses/attribute_mean/{}".format(attribute), torch.mean(l))

        return loss / self.M
