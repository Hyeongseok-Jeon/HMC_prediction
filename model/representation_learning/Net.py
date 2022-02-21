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
        self.Wk = nn.ModuleList([nn.Linear(int(config["n_hidden_after_deconv"]/2), config["n_hidden_after_deconv"]) for i in range(config["max_pred_time"]*config["hz"])])

    def forward(self, trajectory, traj_length):
        # hz2_index.reverse()
        seg_length = []
        for i in range(len(traj_length)):
            hz2_index = []
            j = 0
            while True:
                idx_cand = int(traj_length[i] - 10 / self.config["hz"] * j)
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
        ar_in = self.encoder(enc_in)
        representation = self.autoregressive(ar_in, seg_length)

        cur_time = []
        for i in range(len(seg_length)):
            if seg_length[i] > self.config["max_pred_time"]*self.config["hz"]+1:
                cur_time.append(torch.randint(seg_length[i]-self.config["max_pred_time"]*self.config["hz"]-1, size=(1,)))
            else:
                cur_time.append(torch.randint(1, size=(1,)))
            if i == 0:
                encode_samples = torch.unsqueeze(ar_in[cur_time[i] + 1:cur_time[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1)
                representation_cur = representation[i, cur_time[i],:]
            else:
                representation_cur_tmp = representation[i, cur_time[i],:]
                encode_samples_tmp = torch.unsqueeze(ar_in[sum(seg_length[:i]) + cur_time[i] + 1:sum(seg_length[:i]) + cur_time[i] + self.config["max_pred_time"] * config["hz"] + 1], dim=1)
                representation_cur = torch.cat((representation_cur, representation_cur_tmp), dim=0)
                encode_samples = torch.cat((encode_samples, encode_samples_tmp), dim=1)

        pred = torch.empty((self.config["max_pred_time"]*self.config["hz"], len(seg_length), 256)).float()  # e.g. size 12*8*512
        for i in range(self.config["max_pred_time"]*self.config["hz"]):
            linear = self.Wk[i]
            pred[i] = linear(representation_cur)  # Wk*c_t e.g. size 8*512

        nce = 0
        for i in range(self.config["max_pred_time"]*self.config["hz"]):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * len(seg_length) * (self.config["max_pred_time"]*self.config["hz"])
        accuracy = 1. * correct.item() /len(seg_length)

        return accuracy, nce, hidden