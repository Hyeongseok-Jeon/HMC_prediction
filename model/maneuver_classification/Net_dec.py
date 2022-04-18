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

        self.loss_mean = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, hidden, maneuver_gt, num_per_batch, before_inlet, mode='train'):
        output = self.decoder(hidden)
        maneuver_gt_aug = []
        for i in range(len(num_per_batch)):
            maneuver_cur = maneuver_gt[i:i+1]
            for _ in range(num_per_batch[i]):
                maneuver_gt_aug.append(maneuver_cur)
        maneuver_gt_aug = torch.cat(maneuver_gt_aug)
        label = torch.argmax(maneuver_gt_aug, dim=1)

        loss = self.loss_mean(output, label)
        if mode == 'train':
            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                total = maneuver_gt_aug.size(0)
                correct = (predicted == label).sum().item()

            return loss, total, correct,

        elif mode == 'val':
            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                num_tot = maneuver_gt_aug.size(0)
                correct_tot = (predicted == label).sum().item()
                num_before_inlet = before_inlet.sum().item()
                correct_before_inlet = ((predicted == label) * (before_inlet)).sum().item()
                num_after_inlet = before_inlet.shape[0] - num_before_inlet
                correct_after_inlet = ((predicted == label) * (~before_inlet)).sum().item()
            return num_tot, correct_tot, num_before_inlet, correct_before_inlet, num_after_inlet, correct_after_inlet

