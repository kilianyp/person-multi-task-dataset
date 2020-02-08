import torch
class DummyLoss(torch.nn.Module):
    def __init__(self, endpoint="dummy"):
        super().__init__()
        self.endpoint = endpoint
    def forward(self, x, data):
        x = x[self.endpoint]
        if isinstance(x, list):
            loss = 0
            for i in x:
                loss += i.sum()
            return loss
        else:
            return x.sum()
