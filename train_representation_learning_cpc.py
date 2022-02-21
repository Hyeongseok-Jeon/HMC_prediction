from model.representation_learning.data import pred_loader_1, collate_fn
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.Net import BackBone
from opt.optimizer import ScheduledOptim
import torch
import torch.optim as optim
from koila import LazyTensor, lazy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

dataset = pred_loader_1(config)
dataloader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=True,
                        collate_fn=collate_fn)

model = BackBone(config).cuda()

optimizer = ScheduledOptim(
    optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
    config["n_warmup_steps"])

softmax = torch.nn.Softmax().cuda()
lsoftmax = torch.nn.LogSoftmax().cuda()
timestep = config["max_pred_time"] * config["hz"]

for epoch in range(config["epoch"]):
    for i, data in enumerate(dataloader):
        trajectory, traj_length = data
        trajectory = trajectory.float().cuda()

        accuracy, loss, calc_step = model(trajectory, traj_length)
        print(loss)

        splicing_idx_2 = total_traj.shape[0]
        min_idx = splicing_idx_2 - dataset.config["max_pred_time"] * dataset.config["hz"] - dataset.config["max_hist_time"] * dataset.config["hz"]
        if min_idx < 0:
            min_idx = 0
        splicing_idx_1 = random.sample(range(min_idx, splicing_idx_2), 1)[0]
        observation = total_traj[splicing_idx_1:splicing_idx_2 + 1]


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

        for i in range(len(pred_step)):
            if i == 0:
                pred = model.Wk[pred_step[i]](hist_representation[i:i+1,:])
            else:
                pred_tmp = model.Wk[pred_step[i]](hist_representation[i:i+1,:])
                pred = torch.cat((pred, pred_tmp), dim=0)
        # TODO: 하나의 시퀀스에 대해 여러개의 타임스텝을 하나의 배치로해서 하나의 시퀀스에 대해 여러번 학습하는게 필요한가?
        # 일단은 샘플당 하나의 prediction step에 대해서 학습되도록 되어 있는데
        # 추후에 샘플당 다수의 prediction step을 동시에 학습하는게 필요할수도?
        # pred = torch.empty((self.timestep, batch, 512)).float()  # e.g. size 12*8*512
        # for i in np.arange(0, self.timestep):
        #     linear = self.Wk[i]
        #     pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512

        nce = 0
        total = torch.mm(goal_representation, torch.transpose(pred, 0, 1))  # e.g. size 8*8
        nce += torch.sum(torch.diag(lsoftmax(total)))  # nce is a tensor
        nce /= -1.*config["batch_size"]
        print(nce.item())
        optimizer.zero_grad()
        nce.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()