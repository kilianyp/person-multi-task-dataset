import torch.optim

def build(cfg, params):
    name = cfg['name'].lower()
    # TODO: What if restored optimizer is different?
    init_from_dict = cfg.get('init_from_dict', None)
    if name == "adam":
        lr = cfg['lr']
        betas = cfg['betas']
        eps = cfg['eps']
        weight_decay = cfg['weight_decay']
        amsgrad = cfg['amsgrad']
        return build_adam(params, lr, betas, eps, weight_decay, amsgrad, init_from_dict)
    elif name == 'sgd':
        lr = cfg['lr']
        momentum = cfg['momentum']
        dampening = cfg['dampening']
        weight_decay = cfg['weight_decay']
        nesterov = cfg['nesterov']
        return build_SGD(params, lr, momentum, dampening, weight_decay, nesterov, init_from_dict)

    raise ValueError("Unknown Optimizer: {}".format(name))

def move_opt_to_cuda(opt):
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

def build_adam(params, lr, betas, eps, weight_decay, amsgrad, init_from_dict):

    opt = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay,
                            amsgrad=amsgrad)

    if init_from_dict is not None:
        opt.load_state_dict(init_from_dict)
        move_opt_to_cuda(opt)

    return opt


def build_SGD(params, lr, momentum, dampening, weight_decay, nesterov, init_from_dict):

    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, dampening=dampening,
                           weight_decay=weight_decay, nesterov=nesterov)

    if init_from_dict is not None:
        opt.load_state_dict(init_from_dict)
        move_opt_to_cuda(opt)
    return opt

