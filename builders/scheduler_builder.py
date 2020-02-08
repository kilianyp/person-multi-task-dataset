from schedulers.scheduler import LRScheduler, exponential_decay_fn, multi_step_fn
from functools import partial


def build(cfg, optimizer, last_epoch):

    name = cfg['name']
    if name == "exponential_decay":
        t0 = cfg['t0']
        t1 = cfg['t1']
        eps0 = cfg['lr']
        factor = cfg['factor']
        return _build_exponential_decay(optimizer, t0, t1, eps0, factor, last_epoch)
    elif name == "multi_step":
        lr = cfg['lr']
        milestones = cfg['milestones']
        gamma = cfg['gamma']
        return _build_multi_step(optimizer, last_epoch, lr, gamma, milestones)
    elif name == "test":
        return test(optimizer, last_epoch)
    # TODO choices
    raise ValueError("Schedule {} does not exist. Please choose from {}.".format(name, ", ".join(name)))


def _build_multi_step(optimizer, last_epoch, lr, gamma, milestones):
    if not list(milestones) == sorted(milestones):
        raise ValueError("Milestones should be a list of increasing integers.")
    fn = partial(multi_step_fn, lr=lr, gamma=gamma, milestones=milestones)
    return LRScheduler(fn, optimizer, last_epoch=last_epoch)


def _build_exponential_decay(optimizer, t0, t1, eps0, factor, last_epoch):

    fn = partial(exponential_decay_fn, t0=t0, t1=t1, eps0=eps0, factor=factor)
    return LRScheduler(fn, optimizer, last_epoch)


def test(optimizer, last_epoch):
    def const(t):
        return 2e-4
    return LRScheduler(const, optimizer,
                       last_epoch=last_epoch)
