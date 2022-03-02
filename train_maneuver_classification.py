from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config import config as config_enc
from model.representation_learning.Net import BackBone
from model.maneuver_classification.config import config as config_dec
from model.maneuver_classification.Net import Downstream
import torch
import os
import warnings
import time
import numpy as np

GPU_NUM = config_dec["GPU_id"]
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print('Data list loading ...\n')

file_list = os.listdir(os.getcwd() + '\logs')

print('------------------------------------------------------------')
for i in range(len(file_list)):
    print('File_id : ' + str(file_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

while True:
    s = input('selected target models : ')
    try:
        if int(s) < len(file_list) and int(s) >= 0:
            file_index = int(s)
            file_id = file_list[file_index].split('.')[0]
            break
        else:
            pass
    except:
        pass

ckpt_dir = config_enc['ckpt_dir'] + file_id
ckpt_list = os.listdir(ckpt_dir)
epoch_list = [int(ckpt_list[i].split('_')[1].split('.')[0]) for i in range(len(ckpt_list))]
idx = sorted(range(len(epoch_list)), key=lambda k: epoch_list[k])
ckpt_list = [ckpt_list[idx[i]] for i in range(len(idx))]

print('------------------------------------------------------------')
for i in range(len(ckpt_list)):
    print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

while True:
    s = input('selected target models : ')
    try:
        if int(s) < len(ckpt_list) and int(s) >= 0:
            weight_index = int(s)
            weight = ckpt_list[weight_index]
            break
        else:
            pass
    except:
        pass

warnings.filterwarnings("ignore", category=UserWarning)

# dataset_original = pred_loader_1(config, 'orig')
dataset_train = pred_loader_1(config_dec, 'train', mode='val')
dataset_val = pred_loader_1(config_dec, 'val', mode='val')

dataloader_train = DataLoader(dataset_train,
                              batch_size=config_dec["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val,
                            batch_size=config_dec["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)

encoder = BackBone(config_enc).cuda()
decoder = Downstream(config_dec).cuda()
weights = torch.load(ckpt_dir + '/' + weight)
encoder.load_state_dict(weights['model_state_dict'])

correct_num_tot = 0
full_length_num_tot = 0
loss_tot = 0
loss_calc_num_tot = 0
epoch_time = time.time()

for i, data in enumerate(dataloader_train):
    trajectory, traj_length, conversion, maneuvers = data
    trajectory = trajectory.float().cuda()

    hidden, num_per_batch = encoder(trajectory, traj_length, mode='downstream')
