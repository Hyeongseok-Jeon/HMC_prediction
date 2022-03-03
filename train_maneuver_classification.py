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


optimizer = torch.optim.Adam(decoder.parameters(), lr=config_dec['learning_rate'])

for epoch in range(config_dec['epoch']):
    correct_tot = 0
    calc_tot = 0
    loss_tot = 0
    epoch_time = time.time()
    for i, data in enumerate(dataloader_train):
        trajectory, traj_length, conversion, maneuver_gt = data
        trajectory = trajectory.float().cuda()
        maneuver_gt = torch.cat(maneuver_gt, dim=0).float().cuda()

        hidden, num_per_batch = encoder(trajectory, traj_length, mode='downstream')
        hidden = hidden.detach()
        loss, total, correct = decoder(hidden, maneuver_gt, num_per_batch)

        loss.backward()
        optimizer.step()

        correct_tot += correct
        calc_tot += total
        loss_tot += loss.item()*total

        if data[0].shape[0] == config_dec['batch_size']:
            print('Epoch: %d \t Time: %3.2f sec \t Data: %d/%d \t Loss: %7.5f' % (epoch+1, time.time() - epoch_time, config_dec['batch_size'] * (i + 1), len(dataloader_train.dataset), loss.item()), end='\r')
        else:
            print('Epoch: %d \t Time: %3.2f sec \t Data: %d/%d \t Loss: %7.5f' % (epoch+1, time.time() - epoch_time, config_dec['batch_size'] * i + data[0].shape[0], len(dataloader_train.dataset), loss.item()))

    loss_batch = loss_tot / calc_tot
    logger.info('===> Train Epoch: {} \t Accuracy: {:.2f}%\tLoss: {:.8f}'.format(
        epoch + 1, 100 * correct_tot / calc_tot, loss_batch
    ))