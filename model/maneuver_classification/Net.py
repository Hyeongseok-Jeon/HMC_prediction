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

    def forward(self, hidden, maneuver_gt, num_per_batch, mode='train'):
        output = decoder.decoder(hidden)
        maneuver_gt_aug = []
        for i in range(len(num_per_batch)):
            maneuver_cur = maneuver_gt[i:i+1]
            for _ in range(num_per_batch[i]):
                maneuver_gt_aug.append(maneuver_cur)
        maneuver_gt_aug = torch.cat(maneuver_gt_aug)

        
        return output

