import os
import sys

# root_path = os.path.dirname(os.path.abspath(__file__))

root_path = os.getcwd() + '/model/maneuver_classification/'
sys.path.insert(0, root_path)

import torch
from torch import nn
from modules.decoder import Decoder


class Downstream(nn.Module):
    def __init__(self, config):
        super(Downstream, self).__init__()
        self.config = config
        self.decoder = Decoder(config)
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

    def forward(self, hidden, num_per_batch, mode='train'):
        output = self.decoder(hidden)


        accuracy, nce, torch.sum(batch_idx), calc_step
        return output

