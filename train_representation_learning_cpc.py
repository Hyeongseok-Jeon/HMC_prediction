from model.representation_learning.data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.Net import BackBone
from logger.logger import setup_logs
from opt.optimizer import ScheduledOptim
import torch
import torch.optim as optim
import warnings
import logging
import time

run_name = "maneuver_prediction" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

logger = setup_logs(config['log_dir'], run_name)  # setup logs

warnings.filterwarnings("ignore", category=UserWarning)

dataset = pred_loader_1(config)
dataloader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=True,
                        collate_fn=collate_fn)

model = BackBone(config).cuda()
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info('### Model summary below###\n {}\n'.format(str(model)))
logger.info('===> Model total parameter: {}'.format(model_params))

for i in range(len(config)):
    if i == 0:
        config_log = '                                    ' + list(config.keys())[i] + ': ' + str(list(config.values())[i]) + '\n'
    else:
        config_log = config_log + '                                    ' + list(config.keys())[i] + ': ' + str(list(config.values())[i]) + '\n'

logger.info('===> Configuration parameter\n{}'.format(config_log))

model.train()
optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
    config["n_warmup_steps"])

logger.info('===> Model Training Start')

for epoch in range(config["epoch"]):
    correct_num_tot = 0
    full_length_num_tot = 0
    loss_tot = 0
    loss_calc_num_tot = 0
    for i, data in enumerate(dataloader):
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

        loss_out = loss.item()

    nce_tot = -loss_tot / loss_calc_num_tot
    logger.info('===> Train Epoch: {} \t Accuracy: {:.2f}%\tLoss: {:.8f}'.format(
        epoch, 100 * correct_num_tot / full_length_num_tot, nce_tot
    ))
