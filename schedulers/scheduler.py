from torch.optim.lr_scheduler import _LRScheduler
from logger import get_tensorboard_logger
from bisect import bisect_right


class LRScheduler(_LRScheduler):
    """Learning Rate scheduler.

    TODO only supports single group.
    """

    def __init__(self, schedule_fn, optimizer, last_epoch=-1):
        """
        Args:
            last_epoch: counting from zero
        """
        self.schedule_fn = schedule_fn
        # Manage underlying pytorch behaviour,
        # -1 is special value
        if last_epoch == 0:
            # from scratch
            last_epoch = -1
        else:
            # restored
            last_epoch -= 1
        self.last_epoch = last_epoch
        self.tensorboard_logger = get_tensorboard_logger()
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.schedule_fn(self.last_epoch)
        # TODO Does not support multiple groups
        return [lr] * len(self.optimizer.param_groups)

    def step(self, epoch=None):
        # copy behaviour from _LRScheduler step and add logging
        lr = self.schedule_fn(self.last_epoch + 1)
        self.tensorboard_logger.add_scalar("learning_rate", lr)
        super().step(epoch)


def multi_step_fn(t, lr,  gamma, milestones):
    """A multi step function.
    Copies the behaviour torch.optim.MultiStepLR."""
    return lr * gamma ** bisect_right(milestones, t)


def exponential_decay_fn(t, t0, t1, eps0, factor):
    """Exponentially decays.

    Args:
        t: Current step, starting from 0
        t0: Decay start
        t1: Decay end
        factor: Decay factor
        eps0: start learning rate
    """
    if t < t0:
        return eps0
    lr = eps0 * pow(factor, (t - t0) / (t1 - t0))
    return lr
