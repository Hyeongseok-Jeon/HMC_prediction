from model.representation_learning.data import pred_loader
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.Net import BackBone
from opt.optimizer import ScheduledOptim
import torch
import torch.optim as optim

dataset = pred_loader(config)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

model = BackBone(config).cuda()

optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        config["n_warmup_steps"])

for epoch in range(config["epoch"]):
    for i, data in enumerate(dataloader):
        print(i)
        hist_traj, outlet_state, total_traj, maneuver_index = data
        hist_traj = hist_traj.float().cuda(torch.device('cuda:'+str(config["GPU_id"])))
        outlet_state = outlet_state.float().cuda(torch.device('cuda:'+str(config["GPU_id"])))
        total_traj = total_traj.float().cuda(torch.device('cuda:'+str(config["GPU_id"])))
        maneuver_index = maneuver_index.float().cuda(torch.device('cuda:'+str(config["GPU_id"])))

        hist_representation = model(hist_traj)
        goal_representation, _ = model.encoder(outlet_state)
        goal_representation = model.autoregressive.output(goal_representation)