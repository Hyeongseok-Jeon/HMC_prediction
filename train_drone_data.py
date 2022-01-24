import torch
import os
import tqdm
from model.representation_learning.data import pred_loader
from torch.utils.data import DataLoader

cur_dir = os.getcwd()
data_dir = cur_dir+'/data/drone_data/processed/'

dataset = pred_loader(data_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for i, data in enumerate(dataloader):
    hist_traj, outlet_state, total_traj, maneuver_index = data
    print(i)
