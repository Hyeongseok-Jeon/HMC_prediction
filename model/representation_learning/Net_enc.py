import os
import sys

# root_path = os.path.dirname(os.path.abspath(__file__))

root_path = os.getcwd() + '/model/representation_learning/'
sys.path.insert(0, root_path)

import torch
from torch import nn
from modules.encoder import Encoder
from modules.autoregressive import AutoRegressive
torch.pi = torch.acos(torch.zeros(1)).item() * 2


class BackBone(nn.Module):
    def __init__(self, config):
        super(BackBone, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.autoregressive = AutoRegressive(config)
        self.Wk = nn.ModuleList([nn.Linear(config["n_hidden_after_deconv"], config["n_hidden_after_deconv"]) for i in range(config["max_pred_time"] * config["hz"])])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

    def forward(self, trajectory, traj_length, mode='train', vis=False):
        if mode == 'downstream':
            if vis:
                traj_length_aug = []
                trajectory_aug = []
                for i in range(len(traj_length)):
                    init_index = 0
                    end_index = traj_length[i]
                    traj_tmp = trajectory.clone()
                    origin = traj_tmp[i, end_index - 1, :2]
                    heading = -trajectory[i, end_index - 1, 2]
                    rot = torch.tensor([[torch.cos(torch.deg2rad(heading)), -torch.sin(torch.deg2rad(heading))], [torch.sin(torch.deg2rad(heading)), torch.cos(torch.deg2rad(heading))]], device=trajectory[0].device)
                    traj_tmp[i, :, :2] = traj_tmp[i, :, :2] - origin
                    traj_tmp[i, :, :2] = torch.transpose(torch.mm(rot, torch.transpose(traj_tmp[i, :, :2], 0, 1)), 0, 1)
                    traj_tmp[i, :, 2] = torch.deg2rad(torch.fmod(traj_tmp[i, :, 2] + heading + 720, 360))
                    traj_tmp[i, traj_tmp[i, :, 2] > torch.pi, 2] = traj_tmp[i, traj_tmp[i, :, 2] > torch.pi, 2] - 2 * torch.pi
                    trajectory_aug.append(traj_tmp[i:i + 1, init_index:end_index, :])
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
            else:
                traj_length_aug = []
                trajectory_aug = []
                for i in range(len(traj_length)):
                    init_index = torch.randint(traj_length[i] - 5, size=(1,))
                    end_index = torch.randint(init_index.item() + 5, traj_length[i], size=(1,))
                    traj_tmp = trajectory.clone()
                    origin = traj_tmp[i,end_index-1,:2]
                    heading = -trajectory[i, end_index-1, 2]
                    rot = torch.tensor([[torch.cos(torch.deg2rad(heading)), -torch.sin(torch.deg2rad(heading))], [torch.sin(torch.deg2rad(heading)), torch.cos(torch.deg2rad(heading))]], device=trajectory[0].device)
                    traj_tmp[i, :, :2] = traj_tmp[i, :, :2] - origin
                    traj_tmp[i, :, :2] = torch.transpose(torch.mm(rot, torch.transpose(traj_tmp[i, :, :2], 0, 1)), 0, 1)
                    traj_tmp[i, :, 2] = torch.deg2rad(torch.fmod(traj_tmp[i, :, 2] + heading + 720, 360))
                    traj_tmp[i, traj_tmp[i, :, 2] > torch.pi, 2] = traj_tmp[i, traj_tmp[i, :, 2] > torch.pi, 2] - 2 * torch.pi
                    trajectory_aug.append(traj_tmp[i:i + 1, init_index:end_index, :])
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
                    repres_batch = representation[i, seg_length[i] - 1:seg_length[i], :]
                else:
                    repres_batch_tmp = representation[i, seg_length[i] - 1:seg_length[i], :]
                    repres_batch = torch.cat((repres_batch, repres_batch_tmp), axis=0)

            return repres_batch

        else:
            if mode == 'val':
                seg_length = []
                enc_tot = []
                representation_time_bag = [None for _ in range(11)]
                for i in range(self.config["max_pred_time"] * self.config["hz"], -1, -1):
                    cur_index = trajectory.shape[1]-5*i-1
                    if cur_index < 4:
                        for j in range(self.config["val_augmentation"]):
                            seg_length.append(0)
                        pass
                    else:
                        origin = trajectory[0, cur_index, :2]
                        heading = -trajectory[0, cur_index, 2]
                        rot = torch.tensor([[torch.cos(torch.deg2rad(heading)), -torch.sin(torch.deg2rad(heading))], [torch.sin(torch.deg2rad(heading)), torch.cos(torch.deg2rad(heading))]], device=trajectory[0].device)
                        trajectory_tmp = trajectory.clone()
                        trajectory_tmp[0, :, :2] = trajectory_tmp[0, :, :2] - origin
                        trajectory_tmp[0, :, :2] = torch.transpose(torch.mm(rot, torch.transpose(trajectory_tmp[0, :, :2], 0, 1)), 0, 1)
                        trajectory_tmp[0, :, 2] = torch.deg2rad(torch.fmod(trajectory_tmp[0, :, 2] + heading + 720, 360))
                        trajectory_tmp[0, trajectory_tmp[0, :, 2] > torch.pi, 2] = trajectory_tmp[0, trajectory_tmp[0, :, 2] > torch.pi, 2] - 2 * torch.pi

                        indx_cand = [cur_index - 5 * i for i in range(cur_index + 1) if cur_index - 5 * i > 3]
                        if vis:
                            start_index = indx_cand[-1]
                            for k in range(int((cur_index - start_index) / 5) + 1):
                                tmp = trajectory_tmp[0:1, start_index - 4 + 5 * k:start_index - 4 + 5 * (k + 1), :]
                                if k == 0:
                                    enc_in = tmp
                                else:
                                    enc_in = torch.cat((enc_in, tmp), dim=0)
                            enc_tot.append(enc_in)
                            seg_length.append(enc_in.shape[0])
                        else:
                            for j in range(self.config["val_augmentation"]):
                                start_index = indx_cand[torch.randint(len(indx_cand), size=(1,))]
                                for k in range(int((cur_index - start_index) / 5) + 1):
                                    tmp = trajectory_tmp[0:1, start_index - 4 + 5 * k:start_index - 4 + 5 * (k + 1), :]
                                    noise = torch.normal(0, 0.1, size=(tmp.shape), device=tmp.device)
                                    if k == 0:
                                        enc_in = tmp + noise
                                    else:
                                        enc_in = torch.cat((enc_in, tmp + noise), dim=0)
                                enc_tot.append(enc_in)
                                seg_length.append(enc_in.shape[0])
                enc_tot_t = torch.cat(enc_tot)
                enc_in = torch.transpose(enc_tot_t, 1, 2)
                ar_in = self.encoder(enc_in)
                representation = self.autoregressive(ar_in, seg_length)
                for i in range(11):
                    cur_index = trajectory.shape[1]-5*(10-i)-1
                    if cur_index < 4:
                        pass
                    else:
                        if vis:
                            tmp = representation[i:i + 1, seg_length[i] - 1]
                            representation_time_bag[i] = tmp
                        else:
                            for j in range(self.config["val_augmentation"]):
                                tmp = representation[self.config["val_augmentation"]*i+j:self.config["val_augmentation"]*i+j+1, seg_length[self.config["val_augmentation"]*i+j]-1]
                                if j == 0:
                                    representation_time_bag[i] = tmp
                                else:
                                    representation_time_bag[i] = torch.cat((representation_time_bag[i], tmp), dim=0)

                return representation_time_bag

                # seg_length = []
                # for i in range(len(traj_length)):
                #     hz2_index = []
                #     j = 0
                #     while True:
                #         idx_cand = int(traj_length[i] - 10 / self.config["hz"] * j)
                #         if idx_cand >= 0:
                #             hz2_index.append(idx_cand)
                #             j = j + 1
                #         else:
                #             break
                #     hz2_index.sort()
                #     seg_length.append(len(hz2_index) - 1)
                #     if i == 0:
                #         for k in range(len(hz2_index) - 1):
                #             if k == 0:
                #                 enc_in = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                #             else:
                #                 enc_in_tmp = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                #                 enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                #     else:
                #         for k in range(len(hz2_index) - 1):
                #             enc_in_tmp = trajectory[i:i + 1, hz2_index[k]:hz2_index[k + 1], :]
                #             enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                # enc_in = torch.transpose(enc_in, 1, 2)
                # ar_in = self.encoder(enc_in)
                # representation = self.autoregressive(ar_in, seg_length)
                #
                # encode_samples = ar_in[-1:, :]
                # representations = torch.tensor([representation.shape[1] - time_index - 1 for time_index in range(self.config["max_pred_time"] * self.config["hz"], 0, -1)], device=encode_samples[0].device)
                # pred_steps = torch.tensor([time_index for time_index in range(self.config["max_pred_time"] * self.config["hz"], 0, -1)], device=encode_samples[0].device)
                #
                # representations_mod = representations[representations > -1]
                # pred_steps = pred_steps[representations > -1]
                # representation_cur = representation[0, representations_mod]
                # hist_feature = representation_cur
                #
                # preds = torch.empty((representation_cur.shape[0], len(seg_length), 256), device=encode_samples[0].device).float()  # e.g. size 12*8*512
                # for i in range(representation_cur.shape[0]):
                #     linear = self.Wk[pred_steps[i] - 1]
                #     preds[i] = linear(representation_cur[i])
                #
                # pred = torch.squeeze(preds)
                # target = encode_samples
                # valuable_traj = trajectory[0, hz2_index[0]:]
                #
                # return pred, target, valuable_traj, pred_steps, hist_feature


                # before_inlet_index = torch.where(trajectory[0, :, 0] > 0)[0][0].item()
                # seg_length_before = []
                # seg_length_after = []
                # for i in range(model.config["max_pred_time"] * model.config["hz"]):
                #     for j in range(model.config["val_augmentation"]):
                #         end_indexes_before = torch.randint(4, before_inlet_index + 1, size=(1,))
                #         indx_cand = [(end_indexes_before - 5 * i).item() for i in range(before_inlet_index + 1) if end_indexes_before - 5 * i > 3]
                #         start_indexes_before = indx_cand[torch.randint(len(indx_cand), size=(1,))]
                #
                #         origin = trajectory[0, end_indexes_before, :2]
                #         heading = -trajectory[0, end_indexes_before, 2]
                #         rot = torch.tensor([[torch.cos(torch.deg2rad(heading)), -torch.sin(torch.deg2rad(heading))], [torch.sin(torch.deg2rad(heading)), torch.cos(torch.deg2rad(heading))]], device=trajectory[0].device)
                #         trajectory_tmp = trajectory.clone()
                #         trajectory_tmp[0, :, :2] = trajectory_tmp[0, :, :2] - origin
                #         trajectory_tmp[0, :, :2] = torch.transpose(torch.mm(rot, torch.transpose(trajectory_tmp[0, :, :2], 0, 1)), 0, 1)
                #         trajectory_tmp[0, :, 2] = torch.deg2rad(torch.fmod(trajectory_tmp[0, :, 2] + heading + 720, 360))
                #         trajectory_tmp[0, trajectory_tmp[0, :, 2] > torch.pi, 2] = trajectory_tmp[0, trajectory_tmp[0, :, 2] > torch.pi, 2] - 2 * torch.pi
                #         for k in range(int((end_indexes_before - start_indexes_before) / 5) + 1):
                #             tmp = torch.unsqueeze(trajectory_tmp[0, start_indexes_before - 4 + 5 * k:start_indexes_before - 4 + 5 * (k + 1), :], dim=0)
                #             if k == 0:
                #                 enc_in_before_tmp = tmp
                #             else:
                #                 enc_in_before_tmp = torch.cat((enc_in_before_tmp, tmp), dim=0)
                #         seg_length_before.append(enc_in_before_tmp.shape[0])
                #
                #         end_indexes_after = torch.randint(before_inlet_index, traj_length[0], size=(1,))
                #         indx_cand = [(end_indexes_after - 5 * i).item() for i in range(traj_length[0]) if end_indexes_after - 5 * i > 3]
                #         start_indexes_after = indx_cand[torch.randint(len(indx_cand), size=(1,))]
                #         origin = trajectory[0, end_indexes_after, :2]
                #         heading = -trajectory[0, end_indexes_after, 2]
                #         rot = torch.tensor([[torch.cos(torch.deg2rad(heading)), -torch.sin(torch.deg2rad(heading))], [torch.sin(torch.deg2rad(heading)), torch.cos(torch.deg2rad(heading))]], device=trajectory[0].device)
                #         trajectory_tmp = trajectory.clone()
                #         trajectory_tmp[0, :, :2] = trajectory_tmp[0, :, :2] - origin
                #         trajectory_tmp[0, :, :2] = torch.transpose(torch.mm(rot, torch.transpose(trajectory_tmp[0, :, :2], 0, 1)), 0, 1)
                #         trajectory_tmp[0, :, 2] = torch.deg2rad(torch.fmod(trajectory_tmp[0, :, 2] + heading + 720, 360))
                #         trajectory_tmp[0, trajectory_tmp[0, :, 2] > torch.pi, 2] = trajectory_tmp[0, trajectory_tmp[0, :, 2] > torch.pi, 2] - 2 * torch.pi
                #         for k in range(int((end_indexes_after - start_indexes_after) / 5) + 1):
                #             tmp = torch.unsqueeze(trajectory_tmp[0, start_indexes_after - 4 + 5 * k:start_indexes_after - 4 + 5 * (k + 1), :], dim=0)
                #             if k == 0:
                #                 enc_in_after_tmp = tmp
                #             else:
                #                 enc_in_after_tmp = torch.cat((enc_in_after_tmp, tmp), dim=0)
                #         seg_length_after.append(enc_in_after_tmp.shape[0])
                #
                #         if i == 0 and j == 0:
                #             enc_in_before = enc_in_before_tmp
                #             enc_in_after = enc_in_after_tmp
                #         else:
                #             enc_in_before = torch.cat((enc_in_before, enc_in_before_tmp), dim=0)
                #             enc_in_after = torch.cat((enc_in_after, enc_in_after_tmp), dim=0)
                #
                # seg_length_tot = seg_length_before + seg_length_after
                # enc_in_tot = torch.cat((enc_in_before, enc_in_after), dim=0)
                #
                # enc_in = torch.transpose(enc_in_tot, 1, 2)
                # ar_in = model.encoder(enc_in)
                # representation = model.autoregressive(ar_in, seg_length_tot)
                #
                # representation_before = representation[:int(representation.shape[0]/2)]
                # representation_after = representation[int(representation.shape[0]/2):]
                #
                # representation_before_time = [representation_before[model.config["val_augmentation"]*i:model.config["val_augmentation"]*(i+1),seg_length_before[i]] for i in range(model.config["max_pred_time"] * model.config["hz"])]
                # representation_after_time = [representation_after[model.config["val_augmentation"]*i:model.config["val_augmentation"]*(i+1),seg_length_after[i]] for i in range(model.config["max_pred_time"] * model.config["hz"])]
                # for i in range()
                #
                #
                # return pred, target, valuable_traj, pred_steps, hist_feature

            elif mode == 'train':
                seg_length = []
                cur_times = []
                for i in range(len(traj_length)):
                    if traj_length[i] - 10 * self.config["max_pred_time"] < 5:
                        cur_time = torch.tensor([4])
                    else:
                        cur_time = torch.randint(4, traj_length[i] - 10 * self.config["max_pred_time"], size=(1,))
                    origin = trajectory[i, cur_time[0], :2]
                    heading = -trajectory[i, cur_time[0], 2]
                    rot = torch.tensor([[torch.cos(torch.deg2rad(heading)), -torch.sin(torch.deg2rad(heading))], [torch.sin(torch.deg2rad(heading)), torch.cos(torch.deg2rad(heading))]], device=trajectory[0].device)

                    # plt.scatter(trajectory[i, :, 0].cpu(), trajectory[i, :, 1].cpu())
                    # plt.scatter(origin[0].cpu(), origin[1].cpu())
                    # plt.axis('equal')

                    trajectory[i, :, :2] = trajectory[i, :, :2] - origin
                    trajectory[i, :, :2] = torch.transpose(torch.mm(rot, torch.transpose(trajectory[i, :, :2], 0, 1)), 0, 1)
                    trajectory[i, :, 2] = torch.deg2rad(torch.fmod(trajectory[i, :, 2] + heading + 720, 360))
                    trajectory[i, trajectory[i, :, 2] > torch.pi, 2] = trajectory[i, trajectory[i, :, 2] > torch.pi, 2] - 2 * torch.pi
                    hz2_index = []
                    j = 0
                    while True:
                        idx_cand = int(cur_time - 10 / self.config["hz"] * j)
                        if idx_cand >= 4:
                            hz2_index.append(idx_cand)
                            j = j + 1
                        else:
                            break
                    hz2_index = hz2_index + [cur_time.item() + 5 * (k + 1) for k in range(10)]
                    hz2_index.append(min(hz2_index) - 5)
                    hz2_index.sort()
                    hz2_index = [hz2_index[asdf] for asdf in range(len(hz2_index)) if hz2_index[asdf] < traj_length[i]]
                    cur_times.append(hz2_index.index(cur_time) - 1)
                    if i == 0:
                        for k in range(len(hz2_index) - 1):
                            if k == 0:
                                enc_in = trajectory[i:i + 1, hz2_index[k] + 1:hz2_index[k + 1] + 1, :]
                            else:
                                enc_in_tmp = trajectory[i:i + 1, hz2_index[k] + 1:hz2_index[k + 1] + 1, :]
                                enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                    else:
                        for k in range(len(hz2_index) - 1):
                            enc_in_tmp = trajectory[i:i + 1, hz2_index[k] + 1:hz2_index[k + 1] + 1, :]
                            enc_in = torch.cat((enc_in, enc_in_tmp), dim=0)
                    seg_length.append(len(hz2_index) - 1)

                enc_in = torch.transpose(enc_in, 1, 2)
                ar_in = self.encoder(enc_in)
                representation = self.autoregressive(ar_in, seg_length)

                encode_samples = []
                for i in range(len(seg_length)):
                    if i == 0:
                        encode_samples.append(torch.unsqueeze(ar_in[cur_times[i] + 1:cur_times[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1))
                        representation_cur = representation[i:i + 1, cur_times[i], :]
                    else:
                        representation_cur_tmp = representation[i:i + 1, cur_times[i], :]
                        encode_samples.append(torch.unsqueeze(ar_in[sum(seg_length[:i]) + cur_times[i] + 1:sum(seg_length[:i]) + cur_times[i] + self.config["max_pred_time"] * self.config["hz"] + 1], dim=1))
                        representation_cur = torch.cat((representation_cur, representation_cur_tmp), dim=0)

                pred = torch.empty((self.config["max_pred_time"] * self.config["hz"], len(seg_length), 256), device=encode_samples[0].device).float()  # e.g. size 12*8*512
                for i in range(self.config["max_pred_time"] * self.config["hz"]):
                    linear = self.Wk[i]
                    pred[i] = linear(representation_cur)  # Wk*c_t e.g. size 8*512
                pred_steps = torch.tensor([10 if seg_length[i] > 10 else seg_length[i] - 1 for i in range(len(seg_length))])

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
