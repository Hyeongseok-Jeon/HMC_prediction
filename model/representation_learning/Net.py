import os
import sys
# root_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.getcwd()+'/model/representation_learning/'
sys.path.insert(0, root_path)

import torch
from torch import nn
from modules.encoder import Encoder
from modules.autoregressive import AutoRegressive

class BackBone(nn.Module):
    def __init__(self, config):
        super(BackBone, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.autoregressive = AutoRegressive(config)
        self.Wk = nn.ModuleList([nn.Linear(self.config["n_hidden_after_deconv"], self.config["n_hidden_after_deconv"]) for i in range(self.config["max_pred_time"]*self.config["hz"])])

    def forward(self, trajectory, traj_length):
        # hz2_index.reverse()
        seg_length = []
        for i in range(len(traj_length)):
            hz2_index = []
            j = 0
            while True:
                idx_cand = int(traj_length[i] - 10 / config["hz"] * j)
                if idx_cand >= 0:
                    hz2_index.append(idx_cand)
                    j = j + 1
                else:
                    break
            hz2_index.sort()
            seg_length.append(len(hz2_index)-1)
            if i == 0:
                for k in range(len(hz2_index)-1):
                    if k == 0:
                        enc_in = trajectory[i:i+1, hz2_index[k]:hz2_index[k+1], :]
                    else:
                        enc_in_tmp = trajectory[i:i+1, hz2_index[k]:hz2_index[k+1], :]
                        enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
            else:
                for k in range(len(hz2_index) - 1):
                    enc_in_tmp = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                    enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
        enc_in = torch.transpose(enc_in, 1, 2)
        ar_in = encoder(enc_in)
        representation = autoregressive(ar_in, seg_length)


        return representation

