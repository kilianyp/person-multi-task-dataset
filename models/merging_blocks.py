import torch
import torch.nn as nn

class SingleBlock(nn.Module):
    def calc_output_size(self, dimensions):
        return dimensions[self.endpoint]

    def __init__(self, endpoint):
        super().__init__()
        self.endpoint = endpoint


    def forward(self, outputs):
        # TODO output is also a list
        return torch.cat(outputs[self.endpoint], dim=1)

class ConcatBlock(nn.Module):
    def calc_output_size(self, dimensions):
        return sum([dimensions[endpoint] for endpoint in self.endpoints])

    def __init__(self, endpoints):
        super().__init__()
        self.endpoints = endpoints

    def forward(self, outputs):
        concat_list = []
        for endpoint in self.endpoints:
            emb = outputs[endpoint]
            if isinstance(emb, list):
                concat_list.extend(emb)
            else:
                concat_list.append(emb)

        return torch.cat(concat_list, dim=1)
