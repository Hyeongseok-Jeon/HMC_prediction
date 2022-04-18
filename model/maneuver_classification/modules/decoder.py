import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

from torch import Tensor, nn
from typing import Dict, List, Tuple, Union
import torch

class Decoder(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        linear = []
        for i in range(config["n_linear_layer"]):
            layer = nn.Linear(int(config["n_hidden_after_deconv"]/(2**i)), int(config["n_hidden_after_deconv"]/(2**(i+1))))
            linear.append(layer)
            linear.append(nn.ReLU())
        self.linear = nn.ModuleList(linear)
        self.out = nn.Linear(int(config["n_hidden_after_deconv"]/(2**config["n_linear_layer"])), 4)
        self.softmax = nn.Softmax()

    def forward(self, hidden):
        for i in range(len(self.linear)):
            if i == 0:
                out = self.linear[i](hidden)
            else:
                out = self.linear[i](out)
        out = self.softmax(self.out(out))
        return out