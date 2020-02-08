import torch
from builders import model_builder
import torch.nn as nn

class InferenceModel(object):
    def __init__(self, model_path, augmentation, cuda=True):
        raise NotImplementedError
        self.cuda = cuda

        args = load_args(model_path)
        augment_fn, num_augmentations = augmentation_fn_builder(augmentation)
        self.transform = restore_transform(args, augment_fn)
        model, endpoints = restore_model(args, model_path)
        if self.cuda:
            model = model.cuda()
        self.model = model
        self.model.eval()
        self.endpoints = endpoints
        self.num_augmentations = num_augmentations

    def __call__(self, x):
        with torch.no_grad():
            self.endpoints = self.model(x, self.endpoints)
        return self.endpoints

    def on_pil_list(self, images):
        """Forward pass on an image.
        Args:
            data: A PIL image
        """
        data = []
        for image in images:
            data.append(self.transform(image))
        data = torch.cat(data)
        if self.cuda:
            data = data.cuda()
        with torch.no_grad():
            self.endpoints = self.model(data, self.endpoints)
        #result = self.endpoints["emb"]
        # mean over crops
        # TODO this depends on the data augmentation
        #self.endpoints["emb"] = result.mean(0)
        # COPY otherwise a reference is passed that will be overwritten 
        # by the next forward pass
        return self.endpoints.copy()


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

