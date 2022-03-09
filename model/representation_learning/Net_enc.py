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
        if mode == 'downstream':
            traj_length_aug = []
            trajectory_aug = []
            for i in range(len(traj_length)):
                init_index = torch.randint(traj_length[i] - 5, size=(1,))
                end_index = torch.randint(init_index.item() + 5, traj_length[i], size=(1,))
                trajectory_aug.append(trajectory[i:i + 1, init_index:end_index, :])
                traj_length_aug.append(trajectory_aug[i].shape[1])
            # hz2_index.reverse()
            seg_length = []
            for i in range(len(traj_length_aug)):
                hz2_index = []
                j = 0
                while True:
                    idx_cand = int(traj_length_aug[i] - 10 / self.config["hz"] * j)
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
                            enc_in = trajectory_aug[i][:, hz2_index[k]:hz2_index[k + 1], :]
                        else:
                            enc_in_tmp = trajectory_aug[i][:, hz2_index[k]:hz2_index[k + 1], :]
                            enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                else:
                    for k in range(len(hz2_index) - 1):
                        enc_in_tmp = trajectory_aug[i][:, hz2_index[k]:hz2_index[k + 1], :]
                        enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
            enc_in = torch.transpose(enc_in, 1, 2)
            ar_in = self.encoder(enc_in)
            representation = self.autoregressive(ar_in, seg_length)

            for i in range(len(seg_length)):
                if i == 0:
                    repres_batch = representation[i, :seg_length[i], :]
                else:
                    repres_batch_tmp = representation[i, :seg_length[i], :]
                    repres_batch = torch.cat((repres_batch, repres_batch_tmp), axis=0)

            return repres_batch, seg_length, trajectory_aug

        elif mode == 'lanegcn':
            seg_length = traj_length
            enc_in = torch.transpose(trajectory, 1, 2)
            ar_in = self.encoder(enc_in)
            representation = self.autoregressive(ar_in, seg_length)

            for i in range(len(seg_length)):
                if i == 0:
                    repres_batch = representation[i, seg_length[i]-1:seg_length[i], :]
                else:
                    repres_batch_tmp = representation[i, seg_length[i]-1:seg_length[i], :]
                    repres_batch = torch.cat((repres_batch, repres_batch_tmp), axis=0)

            return repres_batch

        else:
            if mode == 'val':
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
                seg_length = []
                cur_times = []

                for i in range(len(traj_length)):
                    if traj_length[i] - 10 * self.config["max_pred_time"] < 5:
                        cur_time = torch.tensor([4])
                    else:
                        cur_time = torch.randint(4, traj_length[i] - 10 * self.config["max_pred_time"], size=(1,))
                    hz2_index = []
                    j = 0
                    while True:
                        idx_cand = int(cur_time - 10 / self.config["hz"] * j)
                        if idx_cand >= 4:
                            hz2_index.append(idx_cand)
                            j = j + 1
                        else:
                            break
                    hz2_index = hz2_index + [cur_time.item()+5*(k+1) for k in range(10)]
                    hz2_index.append(min(hz2_index)-5)
                    hz2_index.sort()
                    hz2_index = [hz2_index[asdf] for asdf in range(len(hz2_index)) if hz2_index[asdf] < traj_length[i]]
                    cur_times.append(hz2_index.index(cur_time)-1)
                    if i == 0:
                        for k in range(len(hz2_index)-1):
                            if k == 0:
                                enc_in = trajectory[i:i + 1, hz2_index[k]+1:hz2_index[k+1]+1, :]
                            else:
                                enc_in_tmp = trajectory[i:i + 1, hz2_index[k]+1:hz2_index[k+1]+1, :]
                                enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                    else:
                        for k in range(len(hz2_index) - 1):
                            enc_in_tmp = trajectory[i:i + 1, hz2_index[k]+1:hz2_index[k+1]+1, :]
                            enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                    seg_length.append(len(hz2_index)-1)

                enc_in = torch.transpose(enc_in, 1, 2)
                ar_in = self.encoder(enc_in)
                representation = self.autoregressive(ar_in, seg_length)

                encode_samples = []
                for i in range(len(seg_length)):
                    if i == 0:
                        encode_samples.append(torch.unsqueeze(ar_in[cur_times[i]+1:cur_times[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1))
                        representation_cur = representation[i:i+1, cur_times[i], :]
                    else:
                        representation_cur_tmp = representation[i:i+1, cur_times[i], :]
                        encode_samples.append(torch.unsqueeze(ar_in[sum(seg_length[:i]) + cur_times[i] + 1:sum(seg_length[:i]) + cur_times[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1))
                        representation_cur = torch.cat((representation_cur, representation_cur_tmp), dim=0)

                pred = torch.empty((self.config["max_pred_time"] * self.config["hz"], len(seg_length), 256), device=encode_samples[0].device).float()  # e.g. size 12*8*512
                for i in range(self.config["max_pred_time"] * self.config["hz"]):
                    linear = self.Wk[i]
                    pred[i] = linear(representation_cur)  # Wk*c_t e.g. size 8*512
                pred_steps = torch.tensor([10 if seg_length[i] > 10 else seg_length[i]-1 for i in range(len(seg_length))])

                nce = 0
                calc_num = 0
                for i in range(self.config["max_pred_time"] * self.config["hz"]):
                    batch_idx = pred_steps > i
                    if torch.sum(batch_idx) == 0:
                        pass
                    else:
                        total = torch.mm(torch.cat([encode_samples[j][i] for j in range(len(pred_steps)) if batch_idx[j] == True], dim=0), torch.transpose(pred[i, batch_idx], 0, 1))  # e.g. size 8*8
                        calc_num += torch.diag(self.lsoftmax(total)).shape[0]
                        nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
                        correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, torch.sum(batch_idx), device=encode_samples[0].device)))  # correct is a tensor

                nce /= -1. * calc_num
                full_length_num = torch.sum(batch_idx)
                accuracy = 1. * correct.item() / full_length_num

                return accuracy, nce, calc_num, full_length_num
