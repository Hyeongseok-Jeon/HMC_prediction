import os
import glob
import numpy as np
from torch.utils.data import Dataset
import torch
import random


class pred_loader_0(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config["data_dir"]
        self.data_list = config['splicing_num'] * [os.path.basename(x) for x in
                                                   glob.glob(self.data_dir + 'hist_traj/*.npy')]

        self.total_traj_max = 0
        for i in range(len(self.data_list)):
            total_traj = np.load(self.data_dir + 'total_traj/' + self.data_list[i])
            if total_traj.shape[0] > self.total_traj_max:
                self.total_traj_max = total_traj.shape[0]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        hist_traj = np.zeros(shape=(self.hist_traj_max, 3))
        total_traj = np.zeros(shape=(self.total_traj_max, 3))

        hist_traj_raw = np.load(self.data_dir + 'hist_traj/' + self.data_list[0])
        total_traj_raw = np.load(self.data_dir + 'total_traj/' + self.data_list[0])

        splicing_idx = random.sample(range(0, hist_traj_raw.shape[0]), 2)
        hist_traj_raw = hist_traj_raw[min(splicing_idx): max(splicing_idx) + 1]

        for i in range(1, len(hist_traj_raw) - 1):
            rd_num = np.random.rand()
            if rd_num < self.config["occlusion_rate"]:
                hist_traj_raw[i] = -1
        traj_save = hist_traj_raw.copy()

        hist_traj_raw = traj_save.copy()
        if self.config["interpolate"]:
            i = 0
            while i < len(hist_traj_raw) - 1:
                if hist_traj_raw[i, 0] != 0:
                    i = i + 1
                else:
                    for j in range(1, len(hist_traj_raw) - i):
                        if hist_traj_raw[i + j, 0] != 0:
                            next_seen_idx = i + j
                            break
                    for k in range(i, next_seen_idx):
                        hist_traj_raw[k] = hist_traj_raw[i - 1] + (k - i + 1) * (
                                hist_traj_raw[next_seen_idx] - hist_traj_raw[i - 1]) / (next_seen_idx - i + 1)
                    i = i + j

        hist_traj[:hist_traj_raw.shape[0]] = hist_traj_raw
        total_traj[:total_traj_raw.shape[0]] = total_traj_raw
        maneuver_index = np.load(self.data_dir + 'maneuver_index/' + self.data_list[idx])
        outlet_state = np.load(self.data_dir + 'outlet_state/' + self.data_list[idx])
        return hist_traj, outlet_state, total_traj, maneuver_index


class pred_loader_1(Dataset):
    def __init__(self, config, label, mode='train'):
        self.mode = mode
        self.config = config
        if label == 'train':
            self.data_dir = self.config["data_dir_train"]
        elif label == 'val':
            self.data_dir = self.config["data_dir_val"]
        elif label == 'orig':
            self.data_dir = self.config["data_dir_orig"]
        self.data_list = [os.path.basename(x) for x in glob.glob(self.data_dir + 'maneuver_index/*.npy')]
        self.data_list_dup = []
        for x in glob.glob(self.data_dir + 'maneuver_index/*.npy'):
            file_name = os.path.basename(x)
            maneuver = np.load(self.data_dir + 'maneuver_index/' + file_name)
            if (maneuver[1] == 1) or (maneuver[3] == 1):
                for _ in range(config["splicing_num"]):
                    for _ in range(config["LC_multiple"]):
                        self.data_list_dup.append(file_name)
            else:
                for _ in range(config["splicing_num"]):
                    self.data_list_dup.append(file_name)

        self.total_traj_max = 0
        for i in range(len(self.data_list_dup)):
            total_traj = np.load(self.data_dir + 'total_traj/' + self.data_list_dup[i])
            total_traj = total_traj[total_traj[:, 0] > -self.config["FOV"], :]
            if total_traj.shape[0] > self.total_traj_max:
                self.total_traj_max = total_traj.shape[0]

        self.data_distribution_mod = self.get_maneuver_distribution(self.data_list_dup)
        self.data_distribution_origin = self.get_maneuver_distribution(self.data_list)

    def __len__(self):
        return len(self.data_list_dup)

    def __getitem__(self, idx):
        # link_idx = np.load(self.data_dir + 'link_idx/' + self.data_list_dup[idx])
        # maneuver_index = np.load(self.data_dir + 'maneuver_index/' + self.data_list_dup[idx])
        nearest_outlet_state = np.load(self.data_dir + 'nearest_outlet_state/' + self.data_list_dup[idx])
        # outlet_node_state = np.load(self.data_dir + 'outlet_node_state/' + self.data_list_dup[idx])
        total_traj = np.load(self.data_dir + 'total_traj/' + self.data_list_dup[idx])

        outlet_index = np.where(total_traj[:, 0] == nearest_outlet_state[0, 0])[0][0]
        total_traj = total_traj[:outlet_index + 1, :]

        if self.mode == 'train':
            return [total_traj]
        elif self.mode == 'val':
            maneuver_index = np.load(self.data_dir + 'maneuver_index/' + self.data_list_dup[idx])
            conversion = np.load(self.data_dir + 'conversion/' + self.data_list_dup[idx])
            return [total_traj, conversion, maneuver_index]

    def get_maneuver_distribution(self, data):
        maneuver_index_tot = np.zeros(shape=4)
        for i in range(len(data)):
            maneuver_index = np.load(self.data_dir + 'maneuver_index/' + data[i])
            maneuver_index_tot = maneuver_index_tot + maneuver_index
        return maneuver_index_tot


def collate_fn(samples):
    if len(samples[0]) == 1:
        inputs = [torch.from_numpy(i[0]) for i in samples]
        trajectory = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        length = [inputs[i].shape[0] for i in range(len(inputs))]
        return trajectory, length

    else:
        inputs = [torch.from_numpy(i[0]) for i in samples]
        trajectory = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        length = [inputs[i].shape[0] for i in range(len(inputs))]

        conversion = [torch.unsqueeze(torch.from_numpy(i[1]), dim=0) for i in samples]
        maneuver = [torch.unsqueeze(torch.from_numpy(i[2]), dim=0) for i in samples]

        return trajectory, length, conversion, maneuver
