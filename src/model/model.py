import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, hparams):
        pass

    def forward(self, x):
        y_pred = x * 1

        if not self.training:
            # postprocessing steps here
            pass
        
        return y_pred