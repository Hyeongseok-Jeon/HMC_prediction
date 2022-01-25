import os
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

class pred_loader(Dataset):
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

        hist_traj[:hist_traj_raw.shape[0]] = hist_traj_raw
        total_traj[:total_traj_raw.shape[0]] = total_traj_raw

        maneuver_index = np.load(self.data_dir+'maneuver_index/'+self.data_list[idx])
        outlet_state = np.load(self.data_dir+'outlet_state/'+self.data_list[idx])

        return hist_traj, outlet_state, total_traj, maneuver_index