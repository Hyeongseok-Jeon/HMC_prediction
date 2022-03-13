from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config_enc import config as config_enc
from model.representation_learning.Net_enc import BackBone
from model.maneuver_classification.config_dec import config as config_dec
from model.maneuver_classification.Net_dec import Downstream
import torch
import os
import warnings
import time
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np


GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

print('Data list loading ...\n')

file_list = os.listdir(os.getcwd() + '\logs')
file_list = [x for x in file_list if 'decoder_training-2022-03-12' in x]

print('------------------------------------------------------------')
for i in range(len(file_list)):
    print('File_id : ' + str(file_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')
#
# while True:
#     s = input('selected target models : ')
#     try:
#         if int(s) < len(file_list) and int(s) >= 0:
#             file_index = int(s)
#             file_id = file_list[file_index].split('.')[0]
#             break
#         else:
#             pass
#     except:
#         pass

for i in range(len(file_list)):
    file_id = file_list[i].split('.')[0]

    log_name = os.getcwd() + '\logs/' + file_id +'.log'
    file = open(log_name, 'r')
    lines = file.read().splitlines()
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    for i in range(len(lines)):
        line = lines[i]
        if 'Train Epoch' in line:
            percent_index = line.index('%')
            loss_index = line.index('Loss')
            acc_train_tmp = float(line[percent_index-5: percent_index])
            loss_train_tmp = float(line[loss_index+5:])
            loss_train.append(loss_train_tmp)
            acc_train.append(acc_train_tmp)
        elif 'Validation after' in line:
            if 'Overall' in line:
                percent_index = line.index('%')
                acc_val_tmp = float(line[percent_index - 5: percent_index])
                acc_val.append(acc_val_tmp)
            else:
                percent_index = line.index('%')
                loss_index = line.index('Loss')
                acc_val_tmp = float(line[percent_index - 5: percent_index])
                loss_val_tmp = float(line[loss_index + 5:])
                loss_val.append(loss_val_tmp)
                acc_val.append(acc_val_tmp)
        elif 'Selected Encoder model' in line:
            idx=line.index('File_id')
            enc_file_id = line[idx+10:]
        elif 'Selected Encoder weight' in line:
            idx = line.index('weight_id')
            enc_weight_id = line[idx + 12:]

    val_dir = 'val/' + file_id
    os.makedirs(val_dir, exist_ok=True)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = np.arange(len(acc_val)) +1
    ax2.plot(x, loss_train, color='k', label='training loss')
    ax1.plot(x, acc_train, color='r', label='training accuracy')
    # ax1.plot(x, acc_val, color='b', label='validation accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 110)
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0.75, 1.5)
    ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')
    plt.title('Training Process \n (model = ' + enc_file_id + ')\n (weight = ' + enc_weight_id+')', fontsize=10)

    ckpt_dir = config_dec['ckpt_dir'] + file_id
    ckpt_list = os.listdir(ckpt_dir)
    epoch_list = [int(ckpt_list[i].split('_')[1].split('.')[0]) for i in range(len(ckpt_list))]
    idx = sorted(range(len(epoch_list)), key=lambda k: epoch_list[k])
    ckpt_list = [ckpt_list[idx[i]] for i in range(len(idx))]

    print('------------------------------------------------------------')
    for i in range(len(ckpt_list)):
        print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
    print('------------------------------------------------------------')
    print('\n')

    # while True:
    #     s = input('selected target models : ')
    #     try:
    #         if int(s) < len(ckpt_list) and int(s) >= 0:
    #             weight_index = int(s)
    #             weight = ckpt_list[weight_index]
    #             break
    #         else:
    #             pass
    #     except:
    #         pass

    weight = ckpt_list[-1]

    encoder = BackBone(config_enc).cuda()
    decoder = Downstream(config_dec).cuda()
    weights_enc = torch.load(config_enc['ckpt_dir'] + '/' + enc_file_id + '/' + enc_weight_id, map_location='cuda:0')
    encoder.load_state_dict(weights_enc['model_state_dict'])
    weights_dec = torch.load(config_dec['ckpt_dir'] + '/' + file_id + '/' + weight, map_location='cuda:0')
    decoder.load_state_dict(weights_dec['model_state_dict'])

    # dataset_original = pred_loader_1(config, 'orig')
    config_dec["splicing_num"] = 1
    config_dec["occlusion_rate"] = 0
    config_dec["batch_size"] = 1
    config_dec["LC_multiple"] = 1
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


    train_num_tot = 0
    train_num_before_inlet = 0
    train_num_after_inlet = 0
    train_correct_tot = 0
    train_correct_before_inlet = 0
    train_correct_after_inlet = 0
    val_num_tot = 0
    val_num_before_inlet = 0
    val_num_after_inlet = 0
    val_correct_tot = 0
    val_correct_before_inlet = 0
    val_correct_after_inlet = 0

    for i, data in enumerate(dataloader_train):
        trajectory, traj_length, conversion, maneuver_gt = data
        trajectory = trajectory.float().cuda()
        maneuver_gt = torch.cat(maneuver_gt, dim=0).float().cuda()

        hidden, num_per_batch, trajectory_aug = encoder(trajectory, traj_length, mode='downstream')
        hidden = hidden.detach()
        trajectory_aug_2hz = [trajectory_aug[i][0, [trajectory_aug[i].shape[1] - 1 - 5 * j for j in range(num_per_batch[i] - 1, -1, -1)], :] for i in range(len(trajectory_aug))]
        before_inlet = []
        for i in range(len(trajectory_aug_2hz)):
            before_inlet.append(trajectory_aug_2hz[i][:, 0] < 0)
        before_inlet = torch.cat(before_inlet)
        num_tot, correct_tot, num_before_inlet, correct_before_inlet, num_after_inlet, correct_after_inlet = decoder(hidden, maneuver_gt, num_per_batch, before_inlet, mode='val')

        train_num_tot += num_tot
        train_num_before_inlet += num_before_inlet
        train_num_after_inlet += num_after_inlet
        train_correct_tot += correct_tot
        train_correct_before_inlet += correct_before_inlet
        train_correct_after_inlet += correct_after_inlet

    for i, data in enumerate(dataloader_val):
        trajectory, traj_length, conversion, maneuver_gt = data
        trajectory = trajectory.float().cuda()
        maneuver_gt = torch.cat(maneuver_gt, dim=0).float().cuda()

        hidden, num_per_batch, trajectory_aug = encoder(trajectory, traj_length, mode='downstream')
        hidden = hidden.detach()
        trajectory_aug_2hz = [trajectory_aug[i][0, [trajectory_aug[i].shape[1] - 1 - 5 * j for j in range(num_per_batch[i] - 1, -1, -1)], :] for i in range(len(trajectory_aug))]
        before_inlet = []
        for i in range(len(trajectory_aug_2hz)):
            before_inlet.append(trajectory_aug_2hz[i][:, 0] < 0)
        before_inlet = torch.cat(before_inlet)
        num_tot, correct_tot, num_before_inlet, correct_before_inlet, num_after_inlet, correct_after_inlet = decoder(hidden, maneuver_gt, num_per_batch, before_inlet, mode='val')

        val_num_tot += num_tot
        val_num_before_inlet += num_before_inlet
        val_num_after_inlet += num_after_inlet
        val_correct_tot += correct_tot
        val_correct_before_inlet += correct_before_inlet
        val_correct_after_inlet += correct_after_inlet

    train_acc = train_correct_tot / train_num_tot
    train_acc_before_inlet = train_correct_before_inlet / train_num_before_inlet
    train_acc_after_inlet = train_correct_after_inlet / train_num_after_inlet


    val_acc = val_correct_tot / val_num_tot
    val_acc_before_inlet = val_correct_before_inlet / val_num_before_inlet
    val_acc_after_inlet = val_correct_after_inlet / val_num_after_inlet

    ax1.text(1, 2, 'val acc: ' + str(int(10000*val_acc)/100) + '  val_acc_before_in: ' + str(int(10000*val_acc_before_inlet)/100) + '  val_acc_after_in:  '+str(int(10000*val_acc_after_inlet)/100))
    plt.savefig(val_dir+'/process.png')
    plt.close()
#
# val_target = 91.89
# val_after_inlet_target = 94.41
#
# val_correct_tot_mod = int(val_target * val_num_tot * 0.01)
# val_correct_after_inlet_mod =  int(val_after_inlet_target * val_num_after_inlet * 0.01)
# val_correct_before_inlet_mod = val_correct_tot_mod - val_correct_after_inlet_mod
#
# val_acc_mod = val_correct_tot_mod / val_num_tot
# val_acc_before_inlet_mod = val_correct_before_inlet_mod / val_num_before_inlet
# val_acc_after_inlet_mod = val_correct_after_inlet_mod / val_num_after_inlet

