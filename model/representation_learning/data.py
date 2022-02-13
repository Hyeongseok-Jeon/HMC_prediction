import os
import glob
import numpy as np
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt


class pred_loader_0(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config["data_dir"]
        self.data_list = config['splicing_num'] * [os.path.basename(x) for x in glob.glob(self.data_dir +'hist_traj/*.npy')]

        self.hist_traj_max = 0
        self.total_traj_max = 0
        for i in range(len(self.data_list)):
            hist_traj = np.load(self.data_dir+'hist_traj/'+self.data_list[i])
            total_traj = np.load(self.data_dir + 'total_traj/' + self.data_list[i])
            if hist_traj.shape[0] > self.hist_traj_max:
                self.hist_traj_max = hist_traj.shape[0]
            if total_traj.shape[0] > self.total_traj_max:
                self.total_traj_max = total_traj.shape[0]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        hist_traj = np.zeros(shape=(self.hist_traj_max, 3))
        total_traj = np.zeros(shape=(self.total_traj_max, 3))

        hist_traj_raw = np.load(self.data_dir+'hist_traj/'+self.data_list[0])
        total_traj_raw = np.load(self.data_dir+'total_traj/'+self.data_list[0])

        splicing_idx = random.sample(range(0, hist_traj_raw.shape[0]), 2)
        hist_traj_raw = hist_traj_raw[min(splicing_idx): max(splicing_idx)+1]

        for i in range(1,len(hist_traj_raw)-1):
            rd_num = np.random.rand()
            if rd_num < self.config["occlusion_rate"]:
                hist_traj_raw[i] = -1
        traj_save = hist_traj_raw.copy()

        hist_traj_raw = traj_save.copy()
        if self.config["interpolate"]:
            i = 0
            while i < len(hist_traj_raw)-1:
                if hist_traj_raw[i,0] != 0:
                    i = i + 1
                else:
                    for j in range(1, len(hist_traj_raw)-i):
                        if hist_traj_raw[i+j,0] != 0:
                            next_seen_idx = i+j
                            break
                    for k in range(i,next_seen_idx):
                        hist_traj_raw[k] = hist_traj_raw[i-1] + (k-i+1) * (hist_traj_raw[next_seen_idx] - hist_traj_raw[i-1]) / (next_seen_idx-i+1)
                    i = i + j

        hist_traj[:hist_traj_raw.shape[0]] = hist_traj_raw
        total_traj[:total_traj_raw.shape[0]] = total_traj_raw

        maneuver_index = np.load(self.data_dir+'maneuver_index/'+self.data_list[idx])
        outlet_state = np.load(self.data_dir+'outlet_state/'+self.data_list[idx])

        return hist_traj, outlet_state, total_traj, maneuver_index


class pred_loader_1(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_dir = self.config["data_dir"]
        self.data_list = [os.path.basename(x) for x in glob.glob(self.data_dir +'maneuver_index/*.npy')]
        self.data_list_dup = config['splicing_num'] * [os.path.basename(x) for x in glob.glob(self.data_dir +'maneuver_index/*.npy')]


        self.hist_traj_max = 0
        self.total_traj_max = 0
        for i in range(len(self.data_list)):
            hist_traj = np.load(self.data_dir+'hist_traj/'+self.data_list_dup[i])
            total_traj = np.load(self.data_dir + 'total_traj/' + self.data_list_dup[i])
            if hist_traj.shape[0] > self.hist_traj_max:
                self.hist_traj_max = hist_traj.shape[0]
            if total_traj.shape[0] > self.total_traj_max:
                self.total_traj_max = total_traj.shape[0]


    def __len__(self):
        return len(self.data_list_dup)

    def __getitem__(self, idx):
        hist_traj = np.zeros(shape=(self.hist_traj_max, 3))
        total_traj = np.zeros(shape=(self.total_traj_max, 3))

        hist_traj_raw = np.load(self.data_dir+'hist_traj/'+self.data_list_dup[0])
        total_traj_raw = np.load(self.data_dir+'total_traj/'+self.data_list_dup[0])

        splicing_idx = random.sample(range(0, hist_traj_raw.shape[0]), 2)
        hist_traj_raw = hist_traj_raw[min(splicing_idx): max(splicing_idx)+1]

        for i in range(1,len(hist_traj_raw)-1):
            rd_num = np.random.rand()
            if rd_num < self.config["occlusion_rate"]:
                hist_traj_raw[i] = -1
        traj_save = hist_traj_raw.copy()

        hist_traj_raw = traj_save.copy()
        if self.config["interpolate"]:
            i = 0
            while i < len(hist_traj_raw)-1:
                if hist_traj_raw[i,0] != 0:
                    i = i + 1
                else:
                    for j in range(1, len(hist_traj_raw)-i):
                        if hist_traj_raw[i+j,0] != 0:
                            next_seen_idx = i+j
                            break
                    for k in range(i,next_seen_idx):
                        hist_traj_raw[k] = hist_traj_raw[i-1] + (k-i+1) * (hist_traj_raw[next_seen_idx] - hist_traj_raw[i-1]) / (next_seen_idx-i+1)
                    i = i + j

        hist_traj[:hist_traj_raw.shape[0]] = hist_traj_raw
        total_traj[:total_traj_raw.shape[0]] = total_traj_raw

        maneuver_index = np.load(self.data_dir+'maneuver_index/'+self.data_list_dup[idx])
        outlet_state = np.load(self.data_dir+'outlet_state/'+self.data_list_dup[idx])

        return hist_traj, outlet_state, total_traj, maneuver_index

    def get_maneuver_distribution(self):
        maneuver_index_tot = np.zeros(shape=4)
        for i in range(len(data_list)):
            maneuver_index = np.load(data_dir + 'maneuver_index/' + data_list[i])
            maneuver_index_tot = maneuver_index_tot + maneuver_index
            if maneuver_index[0] == 1:
                index = np.load(data_dir + 'link_idx/' + data_list[i])
                print(index)
        outlet_state = np.load(self.data_dir+'outlet_state/'+self.data_list[idx])
