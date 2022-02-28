from model.representation_learning.data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.Net import BackBone
from logger.logger import setup_logs
from opt.optimizer import ScheduledOptim
import torch
import os
import torch.optim as optim
import warnings
import time
import socket
import numpy as np

GPU_NUM = config["GPU_id"]
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

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
        if int(s) < len(file_list) and int(s)>=0:
            file_index = int(s)
            file_id = file_list[file_index].split('.')[0]
            break
        else:
            pass
    except:
        pass

ckpt_dir = config['ckpt_dir'] + file_id
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
        if int(s) < len(ckpt_list) and int(s)>=0:
            weight_index = int(s)
            weight = ckpt_list[weight_index]
            break
        else:
            pass
    except:
        pass

val_dir = 'val/'+file_id+ '/' + weight.split('.')[0]
os.makedirs(val_dir, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# dataset_original = pred_loader_1(config, 'orig')
config["splicing_num"] = 1
config["occlusion_rate"] = 0
config["batch_size"] = 1
dataset_train = pred_loader_1(config, 'train')
dataset_val = pred_loader_1(config, 'val')
dataset_tot = pred_loader_1(config, 'orig')


dataloader_train = DataLoader(dataset_train,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)
dataloader_tot = DataLoader(dataset_tot,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)

model = BackBone(config).cuda()
weights = torch.load(ckpt_dir+'/'+weight)
model.load_state_dict(weights['model_state_dict'])

correct_num_tot = 0
full_length_num_tot = 0
loss_tot = 0
loss_calc_num_tot = 0
epoch_time = time.time()
for i, data in enumerate(dataloader_train):
    trajectory, traj_length = data
    trajectory = trajectory.float().cuda()

    batch_accuracy, loss, full_length_num, loss_calc_num = model(trajectory, traj_length, mode='val')
    loss.backward()
    optimizer.step()
    lr = optimizer.update_learning_rate()

    correct_num_tot += int(batch_accuracy * full_length_num)
    full_length_num_tot += int(full_length_num)
    loss_tot += -loss.item() * loss_calc_num
    loss_calc_num_tot += loss_calc_num

    if data[0].shape[0] == config['batch_size']:
        print('Epoch: %d \t Time: %3.2f sec \t Data: %d/%d \t Loss: %7.5f' % (epoch+1, time.time() - epoch_time, config['batch_size'] * (i + 1), len(dataloader_train.dataset), loss.item()), end='\r')
    else:
        print('Epoch: %d \t Time: %3.2f sec \t Data: %d/%d \t Loss: %7.5f' % (epoch+1, time.time() - epoch_time, config['batch_size'] * i + data[0].shape[0], len(dataloader_train.dataset), loss.item()))

nce_tot = -loss_tot / loss_calc_num_tot
logger.info('===> Train Epoch: {} \t Accuracy: {:.2f}%\tLoss: {:.8f}'.format(
    epoch + 1, 100 * correct_num_tot / full_length_num_tot, nce_tot
))

if (epoch + 1) % config['validataion_period'] == 0:
    model.eval()
    correct_num_tot_val = 0
    full_length_num_tot_val = 0
    loss_tot_val = 0
    loss_calc_num_tot_val = 0
    val_time = time.time()
    for i, data in enumerate(dataloader_val):
        trajectory, traj_length = data
        trajectory = trajectory.float().cuda()
        batch_accuracy, loss, full_length_num, loss_calc_num = model(trajectory, traj_length)
        correct_num_tot_val += int(batch_accuracy * full_length_num)
        full_length_num_tot_val += int(full_length_num)
        loss_tot_val += -loss.item() * loss_calc_num
        loss_calc_num_tot_val += loss_calc_num

    nce_tot_val = -loss_tot_val / loss_calc_num_tot_val
    logger.info('===> Validation after Training epoch: {} \t Accuracy: {:.2f}%\tLoss: {:.8f}'.format(
        epoch + 1, 100 * correct_num_tot_val / full_length_num_tot_val, nce_tot_val
    ))
    model.train()

if (epoch + 1) % config['ckpt_period'] == 0:
    EPOCH = epoch + 1
    PATH = ckpt_dir + "/model_" + str(EPOCH) + ".pt"
    LOSS = nce_tot
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
    }, PATH)
    print('Check point saved: %s' % PATH)
