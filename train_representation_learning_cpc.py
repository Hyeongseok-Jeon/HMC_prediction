from data.drone_data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config_enc import config
from model.representation_learning.Net_enc import BackBone
from logger.logger import setup_logs
from opt.optimizer import ScheduledOptim
import torch
import os
import torch.optim as optim
import warnings
import time
import socket

GPU_NUM = config["GPU_id"]
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

run_name = "maneuver_prediction" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)
ckpt_dir = config['ckpt_dir'] + run_name
os.makedirs(ckpt_dir, exist_ok=True)
logger = setup_logs(config['log_dir'], run_name)  # setup logs

warnings.filterwarnings("ignore", category=UserWarning)

# dataset_original = pred_loader_1(config, 'orig')
dataset_train = pred_loader_1(config, 'train')
dataset_val = pred_loader_1(config, 'val')
dataloader_train = DataLoader(dataset_train,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
dataloader_val = DataLoader(dataset_val,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            collate_fn=collate_fn)

model = BackBone(config).cuda()
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('### Training Machine Ip address ###\n {}\n'.format(socket.gethostbyname(socket.gethostname())))
logger.info('### Model summary below###\n {}\n'.format(str(model)))

for i in range(len(config)):
    if i == 0:
        config_log = '                                    ' + list(config.keys())[i] + ': ' + str(list(config.values())[i]) + '\n'
    else:
        config_log = config_log + '                                    ' + list(config.keys())[i] + ': ' + str(list(config.values())[i]) + '\n'

logger.info('===> Configuration parameter\n{}'.format(config_log))

logger.info('===> Model total parameter: {}'.format(model_params))

model.train()
optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config["learning_rate"], betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
    config["n_warmup_steps"])

logger.info('===> Model Training Start')

for epoch in range(config["epoch"]):
    correct_num_tot = 0
    full_length_num_tot = 0
    loss_tot = 0
    loss_calc_num_tot = 0
    epoch_time = time.time()
    for i, data in enumerate(dataloader_train):
        trajectory, traj_length = data
        trajectory = trajectory.float().cuda()
        optimizer.zero_grad()

        batch_accuracy, loss, full_length_num, loss_calc_num = model(trajectory, traj_length)
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
