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
from logger.logger import setup_logs
import socket

import numpy as np

GPU_NUM = config_dec["GPU_id"]
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

warnings.filterwarnings("ignore", category=UserWarning)
print('Data list loading ...\n')

file_list = os.listdir(os.getcwd() + '/logs')
print('------------------------------------------------------------')
for i in range(len(file_list)):
    print('File_id : ' + str(file_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')
while True:
    s_model = input('selected target models : ')
    try:
        if int(s_model) < len(file_list) and int(s_model) >= 0:
            file_index = int(s_model)
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
print('File_id : Without pretrained encoder', '  File_index : -1')

for i in range(len(ckpt_list)):
    print('File_id : ' + str(ckpt_list[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')

while True:
    s_weight = input('selected target models : ')
    try:
        if int(s_weight) < len(ckpt_list) and int(s_weight) >= -1:
            if int(s_weight) == -1:
                break
            else:
                weight_index = int(s_weight)
                weight = ckpt_list[weight_index]
                break
        else:
            pass
    except:
        pass

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

encoder = BackBone(config_enc).cuda(device)
decoder = Downstream(config_dec).cuda(device)
if int(s_weight) == -1:
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    params_tot = encoder_params + decoder_params

    pass
else:
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    weights = torch.load(ckpt_dir + '/' + weight)
    encoder.load_state_dict(weights['model_state_dict'])

for i in range(len(config_dec)):
    if i == 0:
        config_log = '                                    ' + list(config_dec.keys())[i] + ': ' + str(list(config_dec.values())[i]) + '\n'
    else:
        config_log = config_log + '                                    ' + list(config_dec.keys())[i] + ': ' + str(list(config_dec.values())[i]) + '\n'

if int(s_weight) == -1:
    encoder.train()
decoder.train()

if int(s_weight) == -1:
    params = list(encoder.parameters()) + list(decoder.parameters())
else:
    params = list(decoder.parameters())

optimizer = torch.optim.Adam(params, lr=config_dec['learning_rate'])

if int(s_weight) == -1:
    run_name = "decoder_training_no_pretrained_encoder" + time.strftime("-%Y-%m-%d_%H_%M_%S")
else:
    run_name = "decoder_training" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

if config_dec["logging"]:
    ckpt_dir = config_dec['ckpt_dir'] + run_name
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logs(config_dec['log_dir'], run_name)
    logger.info('### Training Machine Ip address ###\n {}\n'.format(socket.gethostbyname(socket.gethostname())))
    if int(s_weight) == -1:
        weight = 'no_init_weight'
        logger.info('### Model summary below###\n {}\n'.format(str(encoder) + str(decoder)))
        logger.info('===> Configuration parameter\n{}'.format(config_log))
        logger.info('===> Model total parameter: {}'.format(params_tot))
    else:
        logger.info('### Model summary below###\n {}\n'.format(str(decoder)))
        logger.info('===> Configuration parameter\n{}'.format(config_log))
        logger.info('===> Model total parameter: {}'.format(decoder_params))
    logger.info('### Selected Encoder model >>> {}'.format('File_id : ' + str(file_id), '  File_index : ' + str(s_model)))
    logger.info('### Selected Encoder weight >>> {}'.format('weight_id : ' + str(weight), '  File_index : ' + str(s_weight)))
    logger.info('===> Model Training Start')

for epoch in range(config_dec['epoch']):
    correct_tot = 0
    calc_tot = 0
    loss_tot = 0
    epoch_time = time.time()
    for i, data in enumerate(dataloader_train):
        trajectory, traj_length, conversion, maneuver_gt = data
        trajectory = trajectory.float().cuda(device)
        maneuver_gt = torch.cat(maneuver_gt, dim=0).float().cuda(device)

        hidden, num_per_batch, _ = encoder(trajectory, traj_length, mode='downstream')
        if int(s_weight) == -1:
            pass
        else:
            hidden = hidden.detach()
        loss, total, correct = decoder(hidden, maneuver_gt, num_per_batch, None, mode='train')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_tot += correct
        calc_tot += total
        loss_tot += loss.item() * total

        if data[0].shape[0] == config_dec['batch_size']:
            print('Epoch: %d \t Time: %3.2f sec \t Data: %d/%d \t Loss: %7.5f' % (epoch + 1, time.time() - epoch_time, config_dec['batch_size'] * (i + 1), len(dataloader_train.dataset), loss.item()), end='\r')
        else:
            print('Epoch: %d \t Time: %3.2f sec \t Data: %d/%d \t Loss: %7.5f' % (epoch + 1, time.time() - epoch_time, config_dec['batch_size'] * i + data[0].shape[0], len(dataloader_train.dataset), loss.item()))

    loss_batch = loss_tot / calc_tot
    if config_dec["logging"]:
        logger.info('===> Train Epoch: {} \t Accuracy: {:.2f}% \t Loss: {:.8f}'.format(
            epoch + 1, 100 * correct_tot / calc_tot, loss_batch
        ))
    else:
        print('===> Train Epoch: {} \t Accuracy: {:.2f}% \t Loss: {:.8f}'.format(
            epoch + 1, 100 * correct_tot / calc_tot, loss_batch
        ))


    if (epoch + 1) % config_dec['validataion_period'] == 0:
        encoder.eval()
        decoder.eval()
        correct_tot_sum = 0
        num_tot_sum = 0
        correct_before_inlet_sum = 0
        num_before_inlet_sum = 0
        correct_after_inlet_sum = 0
        num_after_inlet_sum = 0
        val_time = time.time()
        for i, data in enumerate(dataloader_val):
            trajectory, traj_length, conversion, maneuver_gt = data
            trajectory = trajectory.float().cuda(device)
            maneuver_gt = torch.cat(maneuver_gt, dim=0).float().cuda(device)

            hidden, num_per_batch, trajectory_aug = encoder(trajectory, traj_length, mode='downstream')
            hidden = hidden.detach()
            trajectory_aug_2hz = [trajectory_aug[i][0,[trajectory_aug[i].shape[1]-1 - 5*j for j in range(num_per_batch[i]-1, -1, -1)],:] for i in range(len(trajectory_aug))]
            before_inlet = []
            for i in range(len(trajectory_aug_2hz)):
                before_inlet.append(trajectory_aug_2hz[i][:,0] < 0)
            before_inlet = torch.cat(before_inlet)
            num_tot, correct_tot, num_before_inlet, correct_before_inlet, num_after_inlet, correct_after_inlet = decoder(hidden, maneuver_gt, num_per_batch, before_inlet, mode='val')

            correct_tot_sum += correct_tot
            num_tot_sum += num_tot
            correct_before_inlet_sum += correct_before_inlet
            num_before_inlet_sum += num_before_inlet
            correct_after_inlet_sum += correct_after_inlet
            num_after_inlet_sum += num_after_inlet

        loss_tot = loss_tot / calc_tot
        if config_dec["logging"]:
            logger.info('===> Validation after Training epoch: {} \t Overall Accuracy: {:.2f}%\t Before inlet accuracy: {:.2f}% \t After inlet accuracy: {:.2f}%'.format(
                epoch + 1, 100 * correct_tot_sum / num_tot_sum, 100 * correct_before_inlet_sum/num_before_inlet_sum, 100 * correct_after_inlet_sum/num_after_inlet_sum
            ))
        else:
            print('===> Validation after Training epoch: {} \t Overall Accuracy: {:.2f}%\t Before inlet accuracy: {:.2f}% \t After inlet accuracy: {:.2f}%'.format(
                epoch + 1, 100 * correct_tot_sum / num_tot_sum, 100 * correct_before_inlet_sum / num_before_inlet_sum, 100 * correct_after_inlet_sum / num_after_inlet_sum
            ))
        encoder.train()
        decoder.train()

    if (epoch + 1) % config_dec['ckpt_period'] == 0:
        EPOCH = epoch + 1
        PATH = ckpt_dir + "/model_" + str(EPOCH) + ".pt"
        LOSS = loss_tot
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
        }, PATH)
        print('Check point saved: %s' % PATH)