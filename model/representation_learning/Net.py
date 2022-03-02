import os
import sys

# root_path = os.path.dirname(os.path.abspath(__file__))

root_path = os.getcwd() + '/model/representation_learning/'
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
        self.Wk = nn.ModuleList([nn.Linear(config["n_hidden_after_deconv"], config["n_hidden_after_deconv"]) for i in range(config["max_pred_time"] * config["hz"])])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

    def forward(self, trajectory, traj_length, mode='train'):
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
            seg_length.append(len(hz2_index) - 1)
            if i == 0:
                for k in range(len(hz2_index) - 1):
                    if k == 0:
                        enc_in = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                    else:
                        enc_in_tmp = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                        enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
            else:
                for k in range(len(hz2_index) - 1):
                    enc_in_tmp = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                    enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
        enc_in = torch.transpose(enc_in, 1, 2)
        ar_in = self.encoder(enc_in)
        representation = self.autoregressive(ar_in, seg_length)

        if mode == 'downstream':
            encode_samples = ar_in[-1:, :]
            num_per_batch = []
            for i in range(len(seg_length)):
                idx_batch = torch.tensor([seg_length[i] - time_index - 1 for time_index in range(self.config["max_pred_time"] * self.config["hz"], 0, -1)], device=encode_samples[0].device)
                idx_batch = idx_batch[idx_batch > -1]
                num_per_batch.append(len(idx_batch))
                if i == 0:
                    repres_batch = representation[i, idx_batch, :]
                else:
                    repres_batch_tmp = representation[i, idx_batch, :]
                    repres_batch = torch.cat((repres_batch, repres_batch_tmp), axis=0)

            return repres_batch, num_per_batch

        elif mode == 'val':
            encode_samples = ar_in[-1:, :]
            representations = torch.tensor([representation.shape[1] - time_index - 1 for time_index in range(self.config["max_pred_time"] * self.config["hz"], 0, -1)], device=encode_samples[0].device)
            pred_steps = torch.tensor([time_index for time_index in range(self.config["max_pred_time"] * self.config["hz"], 0, -1)], device=encode_samples[0].device)

            representations_mod = representations[representations > -1]
            pred_steps = pred_steps[representations > -1]
            representation_cur = representation[0, representations_mod]
            hist_feature = representation_cur

            preds = torch.empty((representation_cur.shape[0], len(seg_length), 256), device=encode_samples[0].device).float()  # e.g. size 12*8*512
            for i in range(representation_cur.shape[0]):
                linear = self.Wk[pred_steps[i] - 1]
                preds[i] = linear(representation_cur[i])

            pred = torch.squeeze(preds)
            target = encode_samples
            valuable_traj = trajectory[0, hz2_index[0]:]

            return pred, target, valuable_traj, pred_steps, hist_feature

        elif mode == 'train':
            cur_time = []
            encode_samples = []
            for i in range(len(seg_length)):
                if seg_length[i] > self.config["max_pred_time"] * self.config["hz"] + 1:
                    cur_time.append(torch.randint(seg_length[i] - self.config["max_pred_time"] * self.config["hz"] - 1, size=(1,)))
                else:
                    cur_time.append(torch.randint(1, size=(1,)))
                if i == 0:
                    encode_samples.append(torch.unsqueeze(ar_in[cur_time[i] + 1:cur_time[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1))
                    representation_cur = representation[i, cur_time[i], :]
                else:
                    representation_cur_tmp = representation[i, cur_time[i], :]
                    encode_samples.append(torch.unsqueeze(ar_in[sum(seg_length[:i]) + cur_time[i] + 1:sum(seg_length[:i]) + cur_time[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1))
                    representation_cur = torch.cat((representation_cur, representation_cur_tmp), dim=0)

            pred = torch.empty((self.config["max_pred_time"] * self.config["hz"], len(seg_length), 256), device=encode_samples[0].device).float()  # e.g. size 12*8*512
            for i in range(self.config["max_pred_time"] * self.config["hz"]):
                linear = self.Wk[i]
                pred[i] = linear(representation_cur)  # Wk*c_t e.g. size 8*512
            pred_steps = seg_length.copy()
            for i in range(len(pred_steps)):
                pred_steps[i] -= 1
            pred_steps = torch.tensor(pred_steps, device=encode_samples[0].device)

            nce = 0
            for i in range(self.config["max_pred_time"] * self.config["hz"]):
                batch_idx = pred_steps > i
                if torch.sum(batch_idx) == 0:
                    pass
                else:
                    total = torch.mm(torch.cat([encode_samples[j][i] for j in range(len(pred_steps)) if batch_idx[j] == True], dim=0), torch.transpose(pred[i, batch_idx], 0, 1))  # e.g. size 8*8
                    nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
                    correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, torch.sum(batch_idx), device=encode_samples[0].device)))  # correct is a tensor

            calc_step = sum([torch.tensor(self.config["max_pred_time"] * self.config["hz"], device=encode_samples[0].device) if pred_steps[i] > self.config["max_pred_time"] * self.config["hz"] - 1 else pred_steps[i] for i in range(len(pred_steps))])
            nce /= -1. * calc_step
            accuracy = 1. * correct.item() / torch.sum(batch_idx)

            return accuracy, nce, torch.sum(batch_idx), calc_step
