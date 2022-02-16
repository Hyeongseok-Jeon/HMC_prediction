from model.representation_learning.data import pred_loader_1
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.Net import BackBone
from opt.optimizer import ScheduledOptim
import torch
import torch.optim as optim
from koila import LazyTensor, lazy

dataset = pred_loader_1(config)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

model = BackBone(config).cuda()

optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
    config["n_warmup_steps"])

loss = torch.nn.L1Loss()

for epoch in range(config["epoch"]):
    for i, data in enumerate(dataloader):
        print(i)
        observation, nearest_outlet_state, maneuver_index, pred_step = data

        max_data_len = 0
        for i in range(observation.shape[0]):
            data_len = torch.where(observation[i, :, 0] == 0)[0][0]
            if data_len > max_data_len:
                max_data_len = data_len
        observation_fit = observation[:, :max_data_len, :]

        observation_fit = observation_fit.float().cuda(torch.device('cuda:' + str(config["GPU_id"])))
        nearest_outlet_state = nearest_outlet_state.float().cuda(torch.device('cuda:' + str(config["GPU_id"])))
        maneuver_index = maneuver_index.float().cuda(torch.device('cuda:' + str(config["GPU_id"])))

        # (observation_fit, nearest_outlet_state) = lazy(observation_fit, nearest_outlet_state, batch=0)

        hist_representation = model(observation_fit)
        goal_point_repre, _ = model.encoder(nearest_outlet_state)

        for i, l in enumerate(model.autoregressive.output):
            if i == 0:
                x = model.autoregressive.output[i](goal_point_repre)
            else:
                x = model.autoregressive.output[i](x)

        goal_representation = torch.squeeze(x)
        loss_val = loss(hist_representation, goal_representation)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
