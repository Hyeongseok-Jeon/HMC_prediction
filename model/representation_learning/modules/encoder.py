import os
import sys

# root_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.getcwd()+'/model/representation_learning/modules'
sys.path.insert(0, root_path)

from torch import nn
import torch
from torch.nn import functional as F
from layers import Res1d, Conv1d


class Encoder(nn.Module):
    """
    Actor feature extractor with Conv1D
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = config["n_hidden_after_deconv"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, enc_in):
        out = enc_in

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            if i == 1:
                out = F.interpolate(out, scale_factor=1.5, mode="linear", align_corners=False)
            elif i == 0:
                out = F.interpolate(out, scale_factor=5/3, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out