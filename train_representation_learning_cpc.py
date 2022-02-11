from model.representation_learning.data import pred_loader
from torch.utils.data import DataLoader
from model.representation_learning.config import config
from model.representation_learning.modules.encoder import Encoder
from model.representation_learning.modules.autoregressive import AutoRegressive

dataset = pred_loader(config)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

encoder = Encoder(config)
autoregressive = AutoRegressive(config)
for i, data in enumerate(dataloader):
    hist_traj, outlet_state, total_traj, maneuver_index = data
    hist_traj = hist_traj.float()
    outlet_state = outlet_state.float()
    total_traj = total_traj.float()
    maneuver_index = maneuver_index.float()

    enc_out, seq_len = encoder(hist_traj)
    print(i)
