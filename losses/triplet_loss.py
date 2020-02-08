import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import get_tensorboard_logger, get_logger
from utils import var2num


def calc_cdist(a, b, metric='euclidean'):
    diff = a[:, None, :] - b[None, :, :]
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(diff*diff, dim=2) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(diff*diff, dim=2)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=2)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)


def calc_cdist_masked(a, b, mask_a, mask_b, metric='euclidean'):
    diff = a[:, None, :] - b[None, :, :]
    mask = mask_a[:, None, :] * mask_b[None, :, :]
    diff = diff * mask
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(diff*diff, dim=2) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(diff*diff, dim=2)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=2)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)



def _apply_margin(x, m):
    if isinstance(m, float):
        return (x + m).clamp(min=0)
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % m)


def batch_hard(cdist, pids, margin):
    """Computes the batch hard loss as in arxiv.org/abs/1703.07737.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
    """
    mask_pos = (pids[None, :] == pids[:, None]).float()

    ALMOST_INF = 9999.9
    furthest_positive = torch.max(cdist * mask_pos, dim=0)[0]
    furthest_negative = torch.min(cdist + ALMOST_INF*mask_pos, dim=0)[0]

    loss = _apply_margin(furthest_positive - furthest_negative, margin)
    return loss


def topk(cdist, pids, k):
    """Calculates the top-k accuracy.

    Args:
        k: k smallest value

    """ 
    with torch.no_grad():
        batch_size = cdist.size()[0]
        if k >= batch_size:
            k = batch_size - 1

        index = torch.topk(cdist, k+1, largest=False, dim=1)[1] #topk returns value and index
        index = index[:, 1:] # drop diagonal

        topk = torch.zeros(cdist.size()[0], device=pids.device).byte()
        topks = []
        for c in index.split(1, dim=1):
            c = c.squeeze() # c is batch_size x 1
            topk = topk | (pids.data == pids[c].data)
            # topk is uint8, this results in a integer division
            acc = torch.sum(topk).float() / float(batch_size)
            topks.append(acc)
        return topks


def active(loss, threshold=1e-5):
    """
    Counts how many samples contribute to the loss.
    Args:
        loss
    Returns:
        The percentage of active samples.
    """
    return (loss > threshold).sum().to(torch.float) / len(loss)


class BatchHardAttention(nn.Module):
    def __init__(self, m, endpoint_name='triplet'):
        super().__init__()
        self.name = "BatchHard(m={})".format(m)
        self.m = m
        self.cdist_fn = calc_cdist_masked
        self.tensorboard_logger = get_tensorboard_logger()
        self.logger = get_logger()
        self.endpoint_name = endpoint_name

    def forward(self, endpoints, data):
        logits = endpoints[self.endpoint_name]
        masks = endpoints['attention_mask']
        pids = data['pid'].to(logits[0].device, non_blocking=True)
        for idx, (logit, mask) in enumerate(zip(logits, masks)):
            cdist = self.cdist_fn(logit, logit, mask, mask)
            loss = batch_hard(cdist, pids, self.m)
            topks = topk(cdist, pids, 5)
            active_triplets = active(loss)
            self.tensorboard_logger.add_scalar("losses/batch_hard_attention/active_{}".format(idx), active_triplets)
            self.tensorboard_logger.add_scalar("losses/batch_hard_attention/min_loss_{}".format(idx), torch.min(loss))
            self.tensorboard_logger.add_scalar("losses/batch_hard_attention/max_loss_{}".format(idx), torch.max(loss))
            self.tensorboard_logger.add_scalar("losses/batch_hard_attention/mean_{}".format(idx), torch.mean(loss))
            self.tensorboard_logger.add_scalar("losses/batch_hard_attention/active_per_logit_{}".format(idx), torch.mean(torch.sum(mask, dim=0)))
            self.tensorboard_logger.add_scalar("losses/batch_hard_attention/active_logits_{}".format(idx), torch.mean(torch.sum(mask, dim=1)))
            self.tensorboard_logger.add_scalar("acc/top-1_{}".format(idx), topks[0])
            self.tensorboard_logger.add_scalar("acc/top-5_{}".format(idx), topks[4])

            self.logger.info(
                    "For %d: "
                    "batch_hard: %f | active: %f |"
                    "acc/top-1: %f | acc/top-5: %f",
                    idx, var2num(torch.mean(loss)), active_triplets, 
                    var2num(topks[0]), var2num(topks[4]))
        return loss

class BatchHard(nn.Module):
    def __init__(self, m, cdist_fn=calc_cdist, endpoint_name='triplet'):
        super().__init__()
        self.name = "BatchHard(m={})".format(m)
        self.m = m
        self.cdist_fn = cdist_fn
        self.tensorboard_logger = get_tensorboard_logger()
        self.logger = get_logger()
        self.endpoint_name = endpoint_name

    def forward(self, endpoints, data):
        logits = endpoints[self.endpoint_name]
        pids = data['pid'].to(logits[0].device, non_blocking=True)
        for idx, logit in enumerate(logits):
            cdist = self.cdist_fn(logit, logit)
            loss = batch_hard(cdist, pids, self.m)
            topks = topk(cdist, pids, 5)
            active_triplets = active(loss)
            self.tensorboard_logger.add_scalar("losses/batch_hard/active_{}".format(idx), active_triplets)
            self.tensorboard_logger.add_scalar("losses/batch_hard/min_loss_{}".format(idx), torch.min(loss))
            self.tensorboard_logger.add_scalar("losses/batch_hard/max_loss_{}".format(idx), torch.max(loss))
            self.tensorboard_logger.add_scalar("losses/batch_hard/mean_{}".format(idx), torch.mean(loss))
            self.tensorboard_logger.add_scalar("acc/top-1_{}".format(idx), topks[0])
            self.tensorboard_logger.add_scalar("acc/top-5_{}".format(idx), topks[4])

            self.logger.info(
                    "For %d: "
                    "batch_hard: %f | active: %f |"
                    "acc/top-1: %f | acc/top-5: %f",
                    idx, var2num(torch.mean(loss)), active_triplets, 
                    var2num(topks[0]), var2num(topks[4]))
        return loss


def batch_soft(cdist, pids, margin, T=1.0):
    """Calculates the batch soft.
    Instead of picking the hardest example through argmax or argmin,
    a softmax (softmin) is used to sample and use less difficult examples as well.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
        T (float): The temperature of the softmax operation.
    """
    # mask where all positivies are set to true
    mask_pos = pids[None, :] == pids[:, None]
    mask_neg = 1 - mask_pos.data

    # only one copy
    cdist_max = cdist.clone()
    cdist_max[mask_neg] = -float('inf')
    cdist_min = cdist.clone()
    cdist_min[mask_pos] = float('inf')

    # NOTE: We could even take multiple ones by increasing num_samples,
    #       the following `gather` call does the right thing!
    idx_pos = torch.multinomial(F.softmax(cdist_max/T, dim=1), num_samples=1)
    idx_neg = torch.multinomial(F.softmin(cdist_min/T, dim=1), num_samples=1)
    positive = cdist.gather(dim=1, index=idx_pos)[:,0]  # Drop the extra (samples) dim
    negative = cdist.gather(dim=1, index=idx_neg)[:,0]
    loss = _apply_margin(positive - negative, margin)
    return loss


class BatchSoft(nn.Module):
    """BatchSoft implementation using softmax.
    
    Also by Tristani as Adaptive Weighted Triplet Loss.
    """

    def __init__(self, m, cdist_fn=calc_cdist, T=1.0, endpoint_name="triplet"):
        """
        Args:
            m: margin
            T: Softmax temperature
        """
        super(BatchSoft, self).__init__()
        self.name = "BatchSoft(m={}, T={})".format(m, T)
        self.m = m
        self.T = T
        self.cdist_fn = cdist_fn
        self.tensorboard_logger = get_tensorboard_logger()
        self.logger = get_logger()

    def forward(self, endpoints, data):
        pids = data['pid']
        logit = endpoints[self.endpoint_name]
        dist = self.cdist_fn(logit, logit)
        loss = batch_soft(dist, pids, self.m, self.T) 
        self.tensorboard_logger.add_scalar("loss/batch_soft/min_loss", torch.min(loss))
        self.tensorboard_logger.add_scalar("loss/batch_soft/max_loss", torch.max(loss))
        self.tensorboard_logger.add_scalar("loss/batch_soft/mean", torch.mean(loss))
        return loss
